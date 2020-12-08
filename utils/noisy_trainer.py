from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import StepLR
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg import vgg19
from datasets.crowd import Crowd, train_val, get_im_list
from losses.full_post_prob import Full_Post_Prob
from losses.full_covar import Full_Cov_Gaussian_Loss 


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class NoisyTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        self.loss = args.loss
        self.skip_test = args.skip_test
        self.add = args.add
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        lists = {}
        train_list = None
        val_list = None
        test_list = None
        lists['train'] = train_list
        lists['val'] = val_list
        lists['test'] = test_list
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x, 
                                  im_list=lists[x]) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                          if x == 'train' else default_collate),
                                          batch_size=(self.args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.datasets['test'] = Crowd(os.path.join(args.data_dir, 'test'),
                                    args.crop_size,
                                    args.downsample_ratio,
                                    args.is_gray, 'val', 
                                    im_list=lists['test'])
        self.dataloaders['test'] = DataLoader(self.datasets['test'],
                                    collate_fn=default_collate,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers*self.device_count,
                                    pin_memory=False)
        self.model = vgg19(down=self.downsample_ratio, bn=args.bn, o_cn=args.o_cn)
        self.model.to(self.device)
        params = list(self.model.parameters()) 
        self.optimizer = optim.Adam(params, lr=args.lr)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Full_Post_Prob(args.sigma,
                                   args.alpha,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device,
                                   add=self.add,
                                   minx=args.minx,
                                   ratio=args.ratio)
        self.criterion = Full_Cov_Gaussian_Loss(args.use_background, self.device, weight=self.args.weight, reg=args.reg)

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.test_flag = False
        self.best_mae = {}
        self.best_mse = {}
        self.best_epoch = {}
        for stage in ['val', 'test']:
            self.best_mae[stage] = np.inf
            self.best_mse[stage] = np.inf
            self.best_epoch[stage] = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch(epoch)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
                if self.test_flag and not self.skip_test:
                    self.val_epoch(stage='test')
                    self.test_flag = False

    def train_eopch(self, epoch=0):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        s_loss = None

        # Iterate over data.
        self.optimizer.zero_grad()
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.autograd.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)

                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs[0]).detach().cpu().numpy()
                res = (pre_count - gd_count[0]) 
                if step % 100 == 0:
                    print(res, pre_count, gd_count[0], loss.item())
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                    time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self, stage='val'):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        epoch_fore = []
        epoch_back = []
        # Iterate over data.
        for inputs, points, name in self.dataloaders[stage]:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                points = points[0].type(torch.LongTensor)
                res = len(points) - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('{} Epoch {}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(stage, self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse[stage] + self.best_mae[stage]):
            self.test_flag = True
            self.best_mse[stage] = mse
            self.best_mae[stage] = mae
            self.best_epoch[stage] = self.epoch 
            logging.info("{} save best mse {:.2f} mae {:.2f} model epoch {}".format(stage,
                                                                            self.best_mse[stage],
                                                                            self.best_mae[stage],
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_{}.pth').format(stage))
        # print log info
        logging.info('Val: Best Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.best_epoch['val'], self.best_mse['val'], self.best_mae['val'], time.time()-epoch_start))
        logging.info('Test: Best Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.best_epoch['test'], self.best_mse['test'], self.best_mae['test'], time.time()-epoch_start))



