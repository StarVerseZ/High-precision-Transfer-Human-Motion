### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
#from models.models import create_model
from models.models import create_gan_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(1)
np.random.seed(1)
# 预设训练时的基本参数
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
#数据载入
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

gan = create_gan_model(opt)

visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, \
        src_fake, trg_fake, \
        src2trg_mask, trg2src_mask, \
        src2trg, trg2src \
        = gan(
            data['src_pose'], # label
            data['src_img'],
            data['src_template'],
            data['trg_pose'], # label
            data['trg_img'],
            data['trg_template'],
            data['src_mask'],
            data['trg_mask'])

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        losses_dict = dict(zip(gan.module.loss_names, losses))

        loss_D_src = losses_dict['D_fake_src'] + losses_dict['D_real_src']
        loss_G_src = losses_dict['G_GAN_src'] + losses_dict['G_GAN_Feat_src'] + losses_dict['G_VGG_src']

        loss_D_trg = losses_dict['D_fake_trg'] + losses_dict['D_real_trg']
        loss_G_trg = losses_dict['G_GAN_trg'] + losses_dict['G_GAN_Feat_trg'] + losses_dict['G_VGG_trg']

        loss_D_A = losses_dict['A_real'] + losses_dict['A_fake']
        loss_A =  losses_dict['A'] + losses_dict['A_VGG'] + losses_dict['A_Feat']

        ############### Backward Pass ####################
        # update generator weights
        gan.module.optimizer_G.zero_grad()
        (loss_G_src + loss_G_trg).backward()
        gan.module.optimizer_G.step()

        gan.module.optimizer_A.zero_grad()
        loss_A.backward()
        gan.module.optimizer_A.step()

        gan.module.optimizer_D_src.zero_grad()
        loss_D_src.backward()
        gan.module.optimizer_D_src.step()

        gan.module.optimizer_D_trg.zero_grad()
        loss_D_trg.backward()
        gan.module.optimizer_D_trg.step()

        gan.module.optimizer_D_A.zero_grad()
        loss_D_A.backward()
        gan.module.optimizer_D_A.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors_gan = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors_gan, t)
            visualizer.plot_current_errors(errors_gan, total_steps)
            visualizer.print_line('')

        ### display output images
        if save_fake:
            src_input = util.tensor2im(data['src_pose'][0][0,:3])
            src_input2 = util.tensor2im(data['src_pose'][0][0,3:])
            src_input[src_input2 != 0] = src_input2[src_input2 != 0]

            trg_input = util.tensor2im(data['trg_pose'][0][0,:3])
            trg_input2 = util.tensor2im(data['trg_pose'][0][0,3:])
            trg_input[trg_input2 != 0] = trg_input2[trg_input2 != 0]

            visuals = OrderedDict([('trg_img', util.tensor2im(data['trg_img'][0][0])),
                                   ('src_img', util.tensor2im(data['src_img'][0][0])),
                                   ('src_input', src_input),
                                   ('trg_input', trg_input),
                                   ('trg_fake', util.tensor2im(trg_fake.data[0])),
                                   ('src_fake', util.tensor2im(src_fake.data[0])),
                                   ('src2trg', util.tensor2im(src2trg.data[0])),
                                   ('src2trg_mask', util.tensor2im(src2trg_mask.data[0])),
                                   ('src_mask', util.tensor2im((data['src_mask'][0].float()*data['src_img'][0]).data[0]))
                                   ])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            gan.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # update lr at end of epoch
    gan.module.update_learning_rate()

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        gan.module.save('latest')
        gan.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
