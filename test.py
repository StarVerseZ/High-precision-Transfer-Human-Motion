### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_gan_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


# test
if not opt.engine and not opt.onnx:
    gan = create_gan_model(opt)
    if opt.data_type == 16:
        gan.half()
    elif opt.data_type == 8:
        gan.type(torch.uint8)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    
    src2trg_mask, src2trg = gan.inference(
        data['src_pose'], 
        data['src_img'], 
        data['trg_template'],
        data['src_mask'])

    # src2trg_mask, src2trg = gan.inference2(
    #     data['trg_pose'], 
    #     data['trg_img'], 
    #     data['src_template'])
    
    # visuals = OrderedDict([('src_img', util.tensor2im(data['src_img'][0])),
    #                        ('src_label', util.tensor2im(data['src_pose'][0])),
    #                        ('src2trg_mask', util.tensor2im(src2trg_mask.data[0])),
    #                        ('src2trg', util.tensor2im(src2trg.data[0]))
    #                        ])
    # src2trg_img = util.tensor2im(src2trg.data[0])
    # src2trg_mask_img = util.tensor2im(src2trg_mask.data[0])
    # img_tensor = torch.cat([src2trg, src2trg_mask], dim=3)
    img_tensor = src2trg_mask
    visuals = OrderedDict([('src2trg_mask', util.tensor2im(img_tensor.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
