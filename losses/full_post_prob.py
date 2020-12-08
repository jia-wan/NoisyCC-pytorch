import torch
from torch.nn import Module
import math
import numpy as np

class Full_Post_Prob(Module):
    def __init__(self, sigma, alpha, c_size, stride, background_ratio, use_background, device, add=False, minx=1e-6, ratio=0.6):
        super(Full_Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma**2
        self.alpha = alpha**2
        self.background_ratio = background_ratio
        self.ratio = ratio
        self.device = device
        self.stride = stride 
        self.add = add
        self.minx = minx
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        
        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis_all = y_dis + x_dis
            dis_all = dis_all.view((dis_all.size(0), -1))

            dis_list = [] 
            i = 0
            for num in num_points_per_image:
                dis_list.append(dis_all[i:i+num,:])
                i += num
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.background_ratio
                        tmp_x = torch.sqrt(min_dis).cpu()
                        std = np.sqrt(self.alpha + self.sigma)
                        bg_dis = (d * torch.exp(-tmp_x/std) )**2  # exp
                        dis = torch.cat([dis, bg_dis.cuda()], 0)  # concatenate background distance to the last

                    tmpm = torch.exp((-0.5/(self.alpha+self.sigma))*dis) / (2*math.pi*(self.alpha+self.sigma))
                    if self.use_bg:
                        tmpv = torch.exp((-0.5/(self.alpha+self.sigma/2))*dis[0:-1]) / (2*math.pi*(self.alpha+self.sigma/2))
                        # sel_v without background point is used for variance ranking and low-rank apprroximation
                        sel_v = ((1.0 / (2.0*math.pi*2.0*self.sigma) * tmpv).sum(0)) - (tmpm[0:-1]**2).sum(0)
                        tmpv = torch.exp((-0.5/(self.alpha+self.sigma/2))*dis) / (2*math.pi*(self.alpha+self.sigma/2))
                        v = ((1.0 / (2.0*math.pi*2.0*self.sigma) * tmpv).sum(0)) - (tmpm[0:-1]**2).sum(0)
                    else:
                        tmpv = torch.exp((-0.5/(self.alpha+self.sigma/2))*dis) / (2*math.pi*(self.alpha+self.sigma/2))
                        v = ((1.0 / (2.0*math.pi*2.0*self.sigma) * tmpv).sum(0)) - (tmpm**2).sum(0)
                        sel_v = v
                    m = tmpm# .sum(0)

                    if self.use_bg:
                        tmp = self.alpha + self.sigma
                        bkg_idx = torch.sqrt(dis[0:-1]) > 2*np.sqrt(tmp)
                    else:
                        tmp = self.alpha + self.sigma
                        bkg_idx = torch.sqrt(dis) > 2*np.sqrt(tmp)
                    
                    # rank v
                    _, inds = torch.sort(sel_v, descending=True)

                    # compute M
                    if self.ratio <= 1:
                        cumv = torch.cumsum(sel_v[inds], dim=0)
                        M = int(1 + torch.sum((cumv / cumv[-1]) <= self.ratio))
                    else:
                        M = int(self.ratio)
                    M = min(M, 100)

                    # select M
                    Minds = inds[0:M]
                    Minds, _ = torch.sort(Minds)

                    # apply regularization
                    MINX = self.minx # 1e-6
                    if self.add:
                        v += MINX
                    else:
                        v = torch.clamp(v, min=MINX)

                    # full conv A
                    dim1 = self.cood.shape[1]
                    dim2 = self.cood.shape[1]
                    xf = (x - self.cood).unsqueeze(1).repeat(1,dim2,1).view(-1, dim1*dim2)
                    yf = (y - self.cood).unsqueeze(2).repeat(1,1,dim1).view(-1, dim1*dim2)
                    xf = xf[:, Minds]
                    yf = yf[:, Minds]
                    qisX = xf.unsqueeze(1) - xf.unsqueeze(2)
                    qisY = yf.unsqueeze(1) - yf.unsqueeze(2)
                    avgX = 0.5*(xf.unsqueeze(1) + xf.unsqueeze(2))
                    avgY = 0.5*(yf.unsqueeze(1) + yf.unsqueeze(2))
                    # the distance between two spatial points
                    df = qisX**2 + qisY**2
                    af = avgX**2 + avgY**2
                    tmpv = (torch.exp((-0.5/(self.sigma*2))*df) / (2*math.pi*(self.sigma*2))) * (torch.exp((-0.5/(self.alpha+self.sigma/2))*af) / (2*math.pi*(self.alpha+self.sigma/2)))
                    tmpv = torch.sum(tmpv, dim=0) + torch.sum(torch.exp(-0.5*df/(self.sigma*2)) / (2*math.pi*(self.sigma*2)), dim=0)
                    if self.use_bg:
                        dis = dis[0:-1]
                    disM = dis[:,Minds]
                    tmpm = torch.exp((-0.5/(self.alpha+self.sigma))*disM) / (2*math.pi*(self.alpha+self.sigma))
                    A = tmpv - torch.mm(tmpm.t(),tmpm)
                    A[range(A.shape[0]), range(A.shape[1])] = 1e-10 

                    # compute B
                    vM = v[Minds] + 1e-10
                    tmpA = vM.unsqueeze(1)*A.inverse()*vM.unsqueeze(0) + torch.diag(vM)
                    B = tmpA.inverse()

                    prob = []
                    prob.append(m)
                    prob.append(v)
                    prob.append(A)
                    prob.append(B)
                    prob.append(Minds)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list


