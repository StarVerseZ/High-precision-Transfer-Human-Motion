### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class GAN(BaseModel):
    def name(self):
        return 'GAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        ##### define networks
        # Generator network
        self.netE_label = networks.define_E(opt.input_nc*2, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global // 2, opt.norm, use_atn=False, use_modes=False, gpu_ids=self.gpu_ids)
        self.netE_template = networks.define_E(opt.output_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global // 2, opt.norm, use_atn=False, use_modes=True, gpu_ids=self.gpu_ids)
        self.netDe = networks.define_De(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global - opt.n_blocks_global // 2, opt.norm, gpu_ids=self.gpu_ids)
        self.netA = networks.define_A(opt.output_nc, opt.output_nc, opt.n_downsample_global, gpu_ids=self.gpu_ids)
        self.netA_face = networks.define_A(opt.output_nc, opt.output_nc, depth=4, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            netD_input_nc = opt.input_nc*2 + opt.output_nc
            self.netD_src = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD_trg = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD_A = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD_A_face = networks.define_D(net_D_input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netE_label, self.name() + 'E_label', opt.which_epoch, pretrained_path)
            self.load_network(self.netA, self.name() + 'A', opt.which_epoch, pretrained_path)
            self.load_network(self.netE_template, self.name() + 'E_template', opt.which_epoch, pretrained_path)
            self.load_network(self.netDe, self.name() + 'De', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD_src, self.name() + 'D_src', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_trg, self.name() + 'D_trg', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_A, self.name() + 'D_A', opt.which_epoch, pretrained_path)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['D_fake_src', 'D_real_src', 'G_GAN_src', 'G_GAN_Feat_src', 'G_VGG_src',
                               'D_fake_trg', 'D_real_trg', 'G_GAN_trg', 'G_GAN_Feat_trg', 'G_VGG_trg',
                               'A_fake', 'A_real', 'A', 'A_Feat', 'A_VGG']

            # initialize optimizers
            # optimizer G
            params = list(self.netE_label.parameters())
            params += list(self.netE_template.parameters())
            params += list(self.netDe.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD_src.parameters())
            self.optimizer_D_src = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D_src = torch.optim.SGD(params, lr=opt.lr, momentum=0.5)

            params = list(self.netD_trg.parameters())
            self.optimizer_D_trg = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D_trg = torch.optim.SGD(params, lr=opt.lr, momentum=0.5)

            params = list(self.netA.parameters())
            self.optimizer_A = torch.optim.Adam(params, lr=opt.lr / 2, betas=(opt.beta1, 0.999))
            #self.optimizer_A = torch.optim.SGD(params, lr=opt.lr, momentum=0.5)

            params = list(self.netD_A.parameters())
            self.optimizer_D_A = torch.optim.Adam(params, lr=opt.lr / 2, betas=(opt.beta1, 0.999))
            #self.optimizer_D_A = torch.optim.SGD(params, lr=opt.lr, momentum=0.5)

    def encode_input(self, inputs):
        for i, input in enumerate(inputs):
            inputs[i] = Variable(input.data.cuda())
        return inputs

    def discriminate(self, netD, label, image):
        input_concat = torch.cat((label, image.detach()), dim=1)
        return netD.forward(input_concat)
    
    def forward(self,
    src_label,
    src_image,
    src_template,
    trg_label,
    trg_image,
    trg_template):

        # Encode Inputs
        src_label, src_image, src_template, trg_label, trg_image, trg_template = \
        self.encode_input([src_label, src_image, src_template, trg_label, trg_image, trg_template])

        # Fake Generation
        src_label_encoded = self.netE_label(src_label)
        trg_label_encoded = self.netE_label(trg_label)

        src_template_encoded = self.netE_template(src_template, 'src')
        trg_template_encoded = self.netE_template(trg_template, 'trg')

        src_fake = self.netDe(src_label_encoded, src_template_encoded, 'src')
        trg_fake = self.netDe(trg_label_encoded, trg_template_encoded, 'trg')

        with torch.no_grad():
            src2trg = self.netDe(src_label_encoded, trg_template_encoded, 'trg')
            trg2src = self.netDe(trg_label_encoded, src_template_encoded, 'src')
        src2trg_mask = self.netA(src_image, src2trg)
        trg2src_mask = self.netA(trg2src, trg_image)

        # Fake Detection and Loss
        pred_fake_src = self.discriminate(
            self.netD_src,
            src_label,
            src_fake)
        loss_D_fake_src = self.criterionGAN(pred_fake_src, False)

        # Real Detection and Loss
        pred_real_src = self.discriminate(
            self.netD_src,
            src_label,
            src_image)
        loss_D_real_src = self.criterionGAN(pred_real_src, True)

        # GAN loss (Fake Passability Loss)
        input_fake_src = torch.cat((
            src_label,
            src_fake), dim=1)
        pred_pass_src = self.netD_src.forward(input_fake_src)
        loss_G_GAN_src = self.criterionGAN(pred_pass_src, True)

        pred_fake_trg = self.discriminate(
            self.netD_trg,
            trg_label,
            trg_fake)
        loss_D_fake_trg = self.criterionGAN(pred_fake_trg, False)

        # Real Detection and Loss
        pred_real_trg = self.discriminate(
            self.netD_trg,
            trg_label,
            trg_image)
        loss_D_real_trg = self.criterionGAN(pred_real_trg, True)

        # GAN loss (Fake Passability Loss)
        input_fake_trg = torch.cat((
            trg_label,
            trg_fake), dim=1)
        pred_pass_trg = self.netD_trg.forward(input_fake_trg)
        loss_G_GAN_trg = self.criterionGAN(pred_pass_trg, True)

        loss_G_GAN_Feat_src = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_src[i])-1):
                    loss_G_GAN_Feat_src += D_weights * feat_weights * \
                        self.criterionFeat(pred_pass_src[i][j], pred_real_src[i][j].detach()) * self.opt.lambda_feat

        loss_G_GAN_Feat_trg = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_trg[i])-1):
                    loss_G_GAN_Feat_trg += D_weights * feat_weights * \
                        self.criterionFeat(pred_pass_trg[i][j], pred_real_trg[i][j].detach()) * self.opt.lambda_feat

        loss_G_VGG_src = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG_src = self.criterionVGG(src_fake, src_image) * self.opt.lambda_feat

        loss_G_VGG_trg = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG_trg = self.criterionVGG(trg_fake, trg_image) * self.opt.lambda_feat
    
        ########
        ########
        ########
        ########

        pred_fake_A = self.discriminate(
            self.netD_A,
            trg_label,
            trg2src_mask)
        loss_A_fake = self.criterionGAN(pred_fake_A, False) 

        pred_real_A = self.discriminate(
            self.netD_A,
            trg_label,
            trg_image)
        loss_A_real = self.criterionGAN(pred_real_A, True)

        pred_pass_trg2src = self.netD_A.forward(torch.cat((
            trg_label,
            trg2src_mask), dim=1))
        loss_A = self.criterionGAN(pred_pass_trg2src, True) 

        loss_A_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_trg[i])-1):
                    loss_A_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_pass_trg2src[i][j], pred_real_A[i][j].detach()) * self.opt.lambda_feat

        loss_A_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_A_VGG = self.criterionVGG(trg2src_mask, trg_image) * self.opt.lambda_feat

        return [loss_D_fake_src, loss_D_real_src, loss_G_GAN_src, loss_G_GAN_Feat_src, loss_G_VGG_src,
                loss_D_fake_trg, loss_D_real_trg, loss_G_GAN_trg, loss_G_GAN_Feat_trg, loss_G_VGG_trg,
                loss_A_fake, loss_A_real, loss_A, loss_A_Feat, loss_A_VGG], \
                src_fake, trg_fake, \
                src2trg_mask, trg2src_mask, \
                src2trg, trg2src

    def inference(self, src_label, src_image, trg_template):
        # Encode Inputs
        src_label, src_image, trg_template = self.encode_input([src_label, src_image, trg_template])

        src_label_encoded = self.netE_label(src_label)
        trg_template_encoded = self.netE_template(trg_template, 'trg')

        src2trg = self.netDe(src_label_encoded, trg_template_encoded, 'trg')
        src2trg_mask = self.netA(src_image, src2trg)

        return src2trg_mask, src2trg

    def inference2(self, src_label, src_image, trg_template):
        # Encode Inputs
        src_label, src_image, trg_template = self.encode_input([src_label, src_image, trg_template])

        src_label_encoded = self.netE_label(src_label)
        trg_template_encoded = self.netE_template(trg_template, 'src')

        src2trg = self.netDe(src_label_encoded, trg_template_encoded, 'src')
        src2trg_mask = self.netA(src_image, src2trg)

        return src2trg_mask, src2trg

    def save(self, which_epoch):
        self.save_network(self.netE_label, self.name() + 'E_label', which_epoch, self.gpu_ids)
        self.save_network(self.netA, self.name() + 'A', which_epoch, self.gpu_ids)
        self.save_network(self.netE_template, self.name() + 'E_template', which_epoch, self.gpu_ids)
        self.save_network(self.netDe, self.name() + 'De', which_epoch, self.gpu_ids)
        self.save_network(self.netD_src, self.name() + 'D_src', which_epoch, self.gpu_ids)
        self.save_network(self.netD_trg, self.name() + 'D_trg', which_epoch, self.gpu_ids)
        self.save_network(self.netD_A, self.name() + 'D_A', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        for param_group in self.optimizer_A.param_groups:
            param_group['lr'] = param_group['lr'] * 0.95
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = param_group['lr'] * 0.95
        print('learning rate updated')

class Inference(GAN):
    def forward(self, src_label, src_image, trg_template):
        return self.inference(src_label, src_image, trg_template)
