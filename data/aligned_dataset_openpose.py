### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import torchvision
import numpy as np
from data.keypoint2img import read_keypoints

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input src labels (label maps)

        dir_src_pose = '_src_openpose'
        self.dir_src_pose = os.path.join(self.root, 'train' + dir_src_pose)
        self.src_openpose_paths = sorted([os.path.join(self.dir_src_pose, fn) for fn in os.listdir(self.dir_src_pose)])

        dir_trg_pose = '_trg_openpose'
        self.dir_trg_pose = os.path.join(self.root, 'train' + dir_trg_pose)
        self.trg_openpose_paths = sorted([os.path.join(self.dir_trg_pose, fn) for fn in os.listdir(self.dir_trg_pose)])

        dir_src_img = '_src_img'
        self.dir_src_img = os.path.join(self.root, 'train' + dir_src_img)
        self.src_img_paths = sorted(make_dataset(self.dir_src_img))

        dir_trg_img = '_trg_img'
        self.dir_trg_img = os.path.join(self.root, 'train' + dir_trg_img)
        self.trg_img_paths = sorted(make_dataset(self.dir_trg_img))

        self.src_size = len(self.src_img_paths) 
        self.trg_size = len(self.trg_img_paths)

        # get face coordinate
    
    def __getitem__(self, index):
        self.rand_src = np.random.randint(self.src_size)
        self.rand_trg = np.random.randint(self.trg_size)

        src_index = index % self.src_size
        src_index_b1 = (index-1) % self.src_size
        src_index_b2 = (index-2) % self.src_size
        trg_index = index % self.trg_size
        trg_index_b1 = (index-1) % self.trg_size
        trg_index_b2 = (index-2) % self.trg_size

        src_img_path = self.src_img_paths[src_index]              
        src_img = Image.open(src_img_path).convert('RGB')
        src_img_path_b1 = self.src_img_paths[src_index_b1]
        src_img_b1 = Image.open(src_img_path_b1).convert('RGB')
        src_img_path_b2 = self.src_img_paths[src_index_b2]
        src_img_b2 = Image.open(src_img_path_b2).convert('RGB')

        trg_img_path = self.trg_img_paths[trg_index]
        trg_img = Image.open(trg_img_path).convert('RGB')
        trg_img_path_b1 = self.trg_img_paths[trg_index_b1]
        trg_img_b1 = Image.open(trg_img_path_b1).convert('RGB')
        trg_img_path_b2 = self.trg_img_paths[trg_index_b2]
        trg_img_b2 = Image.open(trg_img_path_b2).convert('RGB')

        src_template_path = self.src_img_paths[self.rand_src]
        src_template_img = Image.open(src_template_path).convert('RGB')

        trg_template_path = self.trg_img_paths[self.rand_trg]
        trg_template_img = Image.open(trg_template_path).convert('RGB')

        ####
        params = get_params(self.opt, trg_img.size)
        transform = get_transform(self.opt, params)

        src_img_tensor = transform(src_img)
        src_img_tensor_b1 = transform(src_img_b1)
        src_img_tensor_b2 = transform(src_img_b2)

        trg_img_tensor = transform(trg_img)
        trg_img_tensor_b1 = transform(trg_img_b1)
        trg_img_tensor_b2 = transform(trg_img_b2)

        trg_template_tensor = transform(trg_template_img)
        src_template_tensor =  transform(src_template_img)

        ####

        src_pose = self.get_image(self.src_openpose_paths[src_index], trg_img.size, params, input_type='openpose')
        # src_pose = torch.cat((src_densepose, src_openpose))

        src_pose_b1 = self.get_image(self.src_openpose_paths[src_index_b1], trg_img.size, params, input_type='openpose')
        # src_pose_b1 = torch.cat((src_densepose_b1, src_openpose_b1))

        src_pose_b2 = self.get_image(self.src_openpose_paths[src_index_b2], trg_img.size, params, input_type='openpose')
        # src_pose_b2 = torch.cat((src_densepose_b2, src_openpose_b2))

        trg_pose = self.get_image(self.trg_openpose_paths[trg_index], trg_img.size, params, input_type='openpose')
        # trg_pose = torch.cat((trg_densepose, trg_openpose))

        trg_pose_b1 = self.get_image(self.trg_openpose_paths[trg_index_b1], trg_img.size, params, input_type='openpose')
        # trg_pose_b1 = torch.cat((trg_densepose_b1, trg_openpose_b1))

        trg_pose_b2 = self.get_image(self.trg_openpose_paths[trg_index_b2], trg_img.size, params, input_type='openpose')
        # trg_pose_b2 = torch.cat((trg_densepose_b2, trg_openpose_b2))

        if self.opt.phase == 'train':
            path = trg_img_path
        else:
            path = src_img_path

        input_dict = {'src_img': [src_img_tensor,src_img_tensor_b1,src_img_tensor_b2],
                      'trg_img': [trg_img_tensor,trg_img_tensor_b1,trg_img_tensor_b2],
                      'src_pose': [src_pose,src_pose_b1, src_pose_b2], 
                      'trg_pose': [trg_pose,trg_pose_b1, trg_pose_b2],
                      'trg_template': trg_template_tensor, 'src_template': src_template_tensor,
                      'src_mask': [src_mask, src_mask_b1, src_mask_b2], 
                      'trg_mask': [trg_mask, trg_mask_b1, trg_mask_b2],
                      'path': path}

        return input_dict

    def __len__(self):
        if self.opt.phase == 'train':
            return self.trg_size // self.opt.batchSize * self.opt.batchSize
        else:
            return self.src_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

    def get_image(self, A_path, size, params, input_type):
        # if input_type != 'openpose':
        #     A_img = Image.open(A_path).convert('RGB')
        # else:            
        #     random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        #     A_img = Image.fromarray(read_keypoints(A_path, size, random_drop_prob, remove_face_labels=False, basic_point_only=False))            
        A_img = Image.open(A_path).convert('RGB')

        if input_type == 'densepose':
            # randomly remove labels
            A_np = np.array(A_img)
            part_labels = A_np[:,:,2]            
            mask = np.zeros_like(part_labels)
            mask[part_labels == 3] = 1 # left hand
            mask[part_labels == 4] = 1 # right hand
            mask[part_labels == 23] = 1 # left head
            mask[part_labels == 24] = 1 # right head
            import cv2
            mask = cv2.resize(mask, (size[0]//2, size[1]//2))
            mask = np.expand_dims(mask, axis=0)

            for part_id in range(1, 25):
                if (np.random.rand() < self.opt.random_drop_prob):
                    A_np[(part_labels == part_id), :] = 0
            A_img = Image.fromarray(A_np)

        transform_scaleA = get_transform(self.opt, params)
        A_scaled = transform_scaleA(A_img)
        # mask_scaled = transform_scaleA(mask)
        if input_type == 'densepose':
            return A_scaled, mask
        else:
            return A_scaled