import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
from loss import Loss
from util import make_optimizer, calc_psnr, summary


class Operator:
    def __init__(self, config, ckeck_point):
        self.config = config
        self.epochs = config.epochs
        self.uncertainty = config.uncertainty
        self.ckpt = ckeck_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)

        # set model, criterion, optimizer
        self.model = Model(config)
        summary(self.model, config_file=self.ckpt.config_file)

        # set criterion, optimizer
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)

        # load ckpt, model, optimizer
        if self.ckpt.exp_load is not None or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)

    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        train_batch_num = len(data_loader['train'])

        for epoch in range(last_epoch, self.epochs):
            for batch_idx, batch_data in enumerate(data_loader['train']):
                # batch_input, batch_label = batch_data
                batch_input = batch_data['image'].to(self.config.device)
                batch_label = batch_data['label'].to(self.config.device)

                # forward
                batch_results = self.model(batch_input)
                loss = self.criterion(batch_results, batch_label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:5f}'
                      .format(epoch, self.config.epochs,
                              batch_idx, train_batch_num,
                              loss.item()))

                # use tensorboard
                if self.tensorboard:
                    current_global_step = self.ckpt.step()
                    self.summary_writer.add_scalar('train/loss',
                                                   loss, current_global_step)
                    self.summary_writer.add_images("train/input_img",
                                                   batch_input,
                                                   current_global_step)
                    self.summary_writer.add_images("train/mean_img",
                                                   torch.clamp(batch_results['mean'], 0., 1.),
                                                   current_global_step)

            # use tensorboard
            if self.tensorboard:
                print(self.optimizer.get_lr(), epoch)
                self.summary_writer.add_scalar('epoch_lr',
                                               self.optimizer.get_lr(), epoch)

            # test model & save model
            self.optimizer.schedule()
            self.save(self.ckpt, epoch)
            self.val(data_loader)
            self.model.train()

        self.summary_writer.close()

    def val(self, data_loader):
        with torch.no_grad():
            self.model.eval()

            total_psnr = 0.
            psnrs = []
            test_batch_num = len(data_loader['val'])
            for batch_idx, batch_data in enumerate(data_loader['val']):
                batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                # forward
                batch_results = self.model(batch_input)
                current_psnr = calc_psnr(batch_results['mean'], batch_input)
                psnrs.append(current_psnr)
                total_psnr = sum(psnrs) / len(psnrs)
                print("Test iter: {:03d}/{:03d}, Total: {:5f}, Current: {:05f}".format(
                    batch_idx, test_batch_num,
                    total_psnr, psnrs[batch_idx]))

            # use tensorboard
            if self.tensorboard:
                self.summary_writer.add_scalar('val/psnr',
                                               total_psnr, self.ckpt.last_epoch)
                self.summary_writer.add_images("val/input_img",
                                               batch_input, self.ckpt.last_epoch)
                self.summary_writer.add_images("val/mean_img",
                                               torch.clamp(batch_results['mean'], 0., 1.),
                                               self.ckpt.last_epoch)
                if not self.uncertainty == 'normal':
                    self.summary_writer.add_images("val/var_img",
                                                   batch_results['var'],
                                                   self.ckpt.last_epoch)

    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model
        self.optimizer.load(ckpt) # load optimizer

    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt) # save optimizer:


