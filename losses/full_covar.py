from torch.nn.modules import Module
import torch
import torch.nn.functional as F

class Full_Cov_Gaussian_Loss(Module):
    def __init__(self, use_background, device, thr=5, weight=0.01, reg=False, crop=True):
        super(Full_Cov_Gaussian_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background
        self.thr = thr
        self.w = weight
        self.cls_crit = torch.nn.BCELoss()
        self.reg = reg

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        B, C, H, W = pre_density.shape
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                loss += torch.abs(0 - pre_density[idx].sum())
            else:
                N = len(target_list[idx]) 
                m = prob[0] 
                v = prob[1]
                A = prob[2]
                B = prob[3]
                Minds = prob[4]
                
                if self.use_bg:
                    ann = torch.ones_like(m.sum(1))
                    ann[:-1] = target_list[idx]
                    ann[-1] = 0
                    m = (ann.reshape(-1,1)*m).sum(0).view(1,-1)
                else:
                    ann = torch.ones_like(target_list[idx])
                    ann = target_list[idx]
                    m = (ann.reshape(-1,1)*m).sum(0).view(1,-1)

                # normalize 
                factor = N / (m.sum() + 1e-12) 
                m = m * factor
                v = v * (factor)**2
                B = B / (factor)**2
                x = pre_density[idx].reshape(1,C,H,W)
                m = m.reshape(1,1,H,W)
                tmp = x - m 
                tmp = tmp.reshape(C, -1)

                # full covariance loss
                lg1 = torch.sum(tmp**2 / v)
                lg2 = torch.sum(torch.mm(tmp[:,Minds], B)*tmp[:,Minds])
                loss += (0.5 * (lg1-lg2) * self.w)

                # sum=1 regularization
                if self.reg:
                    p = prob[0]/ ((prob[0]).sum(0) + 1e-12) # Nx4096
                    pre_count = torch.sum(pre_density[idx].view(C, 1, -1) * p.unsqueeze(0), dim=2)  # MxN
                    pre_target = torch.ones_like((prob[0]).sum(1))
                    if self.use_bg:
                        pre_target[:-1] = target_list[idx]
                        pre_target[-1] = 0
                    reg_loss = torch.sum(torch.abs(pre_count-pre_target.reshape(1,-1)))
                    loss += reg_loss 


        loss = loss / len(prob_list)
        return loss



