import traceback
import random
from random import choice

import os
import torch
from torch.utils.data import Dataset, Subset

import numpy as np
import cv2
import json
import skimage.io as io
from PIL import Image, ImageFilter
from torchvision.transforms import functional as func_transforms
#import torchvision

import tqdm
import pickle

import torch.nn as nn

import warnings

from utils import imgtrans
from utils import handutils
#from data.fhbhands import FHBHands,HandDataset_fhb
from utils.fh_utils import proj_func

import copy
import imageio
from pytorch3d.io import load_obj
from manotorch.manolayer import ManoLayer
from utils.DARTset_utils import (aa_to_rotmat, fit_ortho_param, ortho_project,
                           plot_hand, rotmat_to_aa)

def pickle_load(path):
    with open(path, 'rb') as fi:
        output = pickle.load(fi)
    return output

def get_dataset(
    dat_name,
    set_name,
    base_path,
    queries,
    use_cache=True,
    limit_size=None,
    train = False,
    split = None,
    if_use_j2d: bool = False
):
    if dat_name == "FreiHand":
        pose_dataset = FreiHand(
            base_path=base_path,
            set_name = set_name)
        sides = 'right',
    elif dat_name == "Dart":
        if set_name == 'training':
            set_name = 'train'
        else:
            set_name = 'test'
        
        pose_dataset = DARTset(
            data_root = base_path,
            data_split = set_name)
        sides = 'right'
        
    elif dat_name == 'RHD':
        pose_dataset = RHD(
            base_path=base_path,
            set_name = set_name,)
        sides = 'both'

    elif dat_name == 'HO3D':
        pose_dataset = HO3D(
            base_path=base_path,
            set_name = set_name,)
        sides = 'right'
    else:
        print("not supported dataset.")
        return
    
    dataset = HandDataset(
        dat_name,
        pose_dataset,
        queries=queries,
        sides = sides,
        #directory=None,
        is_train=train,
        #set_name=None,
        if_use_j2d = if_use_j2d
    )
    
    if limit_size is not None:
        if len(dataset) < limit_size:
            warnings.warn(
                "limit size {} > dataset size {}, working with full dataset".format(
                    limit_size, len(dataset)
                )
            )
        else:
            print( "Working wth subset of {} of size {}".format(dat_name, limit_size))
            dataset = Subset(dataset, list(range(limit_size)))
    return dataset

class HandDataset(Dataset):
    '''
        Get samples after processing given queries.
    '''
    def __init__(
        self, 
        dat_name,
        pose_dataset, 
        if_use_j2d: bool,
        #directory=None, 
        is_train=None, 
        #set_name=None,
        queries = None,
        sides="both",
        center_idx=9,
        #queries = ['images','masks','Ks','scales','manos','joints'],
    ):
        self.pose_dataset = pose_dataset
        #self.set_name = set_name
        self.dat_name = dat_name

        self.queries = queries
        
        self.sides = sides
        self.train = is_train
        self.inp_res = 320
        self.inp_res1 = 224# For crop 
        self.center_idx = center_idx
        self.center_jittering = 0.2
        self.scale_jittering = 0.3
        self.max_rot = np.pi
        self.blur_radius = 0.5
        self.brightness = 0.3#0.5
        self.saturation = 0.3#0.5
        self.hue = 0.15
        self.contrast = 0.5
        self.block_rot = False
        self.black_padding = False
        self.data_pre = False
        self.if_use_j2d = if_use_j2d
    
    def __len__(self):
        return len(self.pose_dataset)
    
    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}
        #sample['dataset']=self.dat_name

        # Freihand
        if self.dat_name == 'FreiHand':
            image = self.pose_dataset.get_img(idx)
            if 'images' in query:
                sample['images']=func_transforms.to_tensor(image).float()#image
            
            if 'bgimgs' in query:
                bgimg = self.pose_dataset.get_bgimg()
                sample['bgimgs']=func_transforms.to_tensor(bgimg).float()
            
            if 'refhand' in query:
                refhand = self.pose_dataset.get_refhand()
                sample['refhand']=func_transforms.to_tensor(refhand).float()

            if 'maskRGBs' in query:
                maskRGB = self.pose_dataset.get_maskRGB(idx)
                sample['maskRGBs']=maskRGB
            
            K = self.pose_dataset.get_K(idx)
            if 'Ks' in query:
                #K_0 = torch.zeros(3,1)
                #K = torch.cat((K,K_0),dim=1).float()
                sample['Ks']=torch.FloatTensor(K)#K
            if 'scales' in query:
                scale = self.pose_dataset.get_scale(idx)
                sample['scales']=scale
            if 'manos' in query:
                mano = self.pose_dataset.get_mano(idx)
                sample['manos']=mano
            if 'joints' in query or 'trans_joints' in query:
                joint = self.pose_dataset.get_joint(idx)
                if 'joints' in query:
                    sample['joints']=joint
            if 'verts' in query or 'trans_verts' in query:
                verts = self.pose_dataset.get_vert(idx)
                if 'verts' in query:
                    sample['verts']=verts
            if 'open_2dj' in query or 'trans_open_2dj' in query:
                open_2dj = self.pose_dataset.get_open_2dj(idx)
                if 'open_2dj' in query:
                    sample['open_2dj']=open_2dj
                open_2dj_con = self.pose_dataset.get_open_2dj_con(idx)
                sample['open_2dj_con']=open_2dj_con
            if 'cv_images' in query:
                cv_image = self.pose_dataset.get_cv_image(idx)
                sample['cv_images']=cv_image
            sample['idxs']=idx
            if 'masks' in query or 'trans_masks' in query:
                if idx >= 32560:
                    idx_this = idx % 32560#check
                else:
                    idx_this = idx
                mask = self.pose_dataset.get_mask(idx_this)
                if 'masks' in query:
                    sample['masks']=torch.round(func_transforms.to_tensor(mask))#mask
            if 'CRFmasks' in query:
                if idx >= 32560:
                    idx_this = idx % 32560#check
                else:
                    idx_this = idx
                CRFmask = self.pose_dataset.get_CRFmask(idx_this)
                sample['CRFmasks']=torch.round(func_transforms.to_tensor(CRFmask))#CRFmask
            # augmentated results
            if self.train:
                if 'trans_images' in query:
                    center = np.asarray([112, 112])
                    scale = 224
                    # Scale jittering
                    '''
                    scale_jittering = self.scale_jittering * np.random.randn() + 1
                    scale_jittering = np.clip(
                        scale_jittering,
                        1 - self.scale_jittering,
                        1 + self.scale_jittering,
                    )
                    scale = scale * scale_jittering
                    '''

                    # Random rotations
                    rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
                    rot_mat = np.array(
                        [
                            [np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1],
                        ]
                    ).astype(np.float32)
                    affinetrans, post_rot_trans = handutils.get_affine_transform(
                        center, scale, [224, 224], rot=rot
                    )
                    trans_images = handutils.transform_img(
                        image, affinetrans, [224, 224]
                    )
                    sample['trans_images'] = func_transforms.to_tensor(trans_images).float()
                    sample['post_rot_trans']=post_rot_trans
                    sample['rot'] = rot
                if 'trans_open_2dj' in query:
                    trans_open_j2d = handutils.transform_coords(open_2dj.numpy(),affinetrans)
                    sample['trans_open_2dj']=torch.from_numpy(np.array(trans_open_j2d)).float()
                if 'trans_Ks' in query:
                    trans_Ks = post_rot_trans.dot(K)
                    sample['trans_Ks']=torch.from_numpy(trans_Ks).float()
                if 'trans_CRFmasks' in query:
                    trans_CRFmasks = handutils.transform_img(
                        CRFmask, affinetrans, [224, 224]
                    )
                    sample['trans_CRFmasks']=torch.round(func_transforms.to_tensor(trans_CRFmasks))
                if 'trans_masks' in query:
                    trans_masks = handutils.transform_img(
                        mask, affinetrans, [224, 224]
                    )
                    sample['trans_masks']=torch.round(func_transforms.to_tensor(trans_masks))
                if 'trans_joints' in query:
                    trans_joint = rot_mat.dot(
                        joint.transpose(1, 0)
                    ).transpose()
                    sample['trans_joints'] = torch.from_numpy(trans_joint)
                if 'trans_verts' in query:
                    trans_verts = rot_mat.dot(
                        verts.transpose(1, 0)
                    ).transpose()
                    sample['trans_verts'] = torch.from_numpy(trans_verts)
                #sample['rot_mat'] = torch.from_numpy(rot_mat)
            if self.if_use_j2d:
                if 'images' in sample:
                    assert 'open_2dj' in sample, "You should include 'open_2dj' in queries to use it as input."
                    sample['images'] = torch.cat([sample['images'], sample['open_2dj']], dim=0)
                if 'trans_images' in sample:
                    assert 'trans_open_2dj' in sample, "You should include 'trans_open_2dj' in queries to use it as input."
                    print(sample['trans_images'].shape, sample['trans_open_2dj'].shape )
                    sample['trans_images'] = torch.cat([sample['trans_images'], sample['trans_open_2dj']], dim=0)
        # RHD
        if self.dat_name == 'RHD':
            #print("-------------------------")
            # Get original image
            '''
            if 'base_images' in query or 'trans_images' in query:
                center, scale = self.pose_dataset.get_center_scale(idx)
                needs_center_scale = True
                image = self.pose_dataset.get_img(idx)
                if 'base_images' in query:
                    sample['images']=func_transforms.to_tensor(image).float()
            else:
                needs_center_scale = False
            '''
            

            image = self.pose_dataset.get_img(idx)
            
            if 'base_images' in query:
                sample['images']=func_transforms.to_tensor(image).float()
            
            mask_ori = self.pose_dataset.get_mask(idx)#[320,320]
            mask = (func_transforms.to_tensor(mask_ori)*255).int()
            
            if 'base_masks' in query:
                sample['masks']=mask
            
            #needs_center_scale=False
            mask_r = (mask>17)
            # mask_l = (mask>1) - mask_r
            mask_l = (mask > 1) & (~mask_r)
            #one_map, zero_map = torch.ones_like(mask), torch.zeros_like(mask)
            num_px_left_hand = torch.sum(mask_l)
            num_px_right_hand = torch.sum(mask_r)

            sample['mask_r']=mask_r
            sample['mask_l']=mask_l
            sample['num_l']=num_px_left_hand
            sample['num_r']=num_px_right_hand
            joints2d_vis = self.pose_dataset.get_j2dvis(idx)
            joints2d_vis_l = joints2d_vis[:21]
            joints2d_vis_r = joints2d_vis[21:]

            vis_l_num = joints2d_vis_l.int().sum()
            vis_r_num = joints2d_vis_r.int().sum()

            if vis_r_num < vis_l_num:
                hand_side = torch.zeros(1).int()
            elif vis_l_num < vis_r_num:
                hand_side = torch.ones(1).int()
            else:
                hand_side = torch.where(num_px_left_hand>num_px_right_hand,torch.zeros(1).int(),torch.ones(1).int())
            # 0 left; 1 right
            if 'sides' in query:
                sample['sides'] = hand_side
            
            joints2d = self.pose_dataset.get_joints2d(idx)
            joints2d = torch.from_numpy(joints2d)
            '''
            if 'base_joints2d' in query:
                sample['joints2d'] = joints2d
            '''
            
            '''
            if 'joints2d_vis' in query:
                sample['joints2d_vis']=joints2d_vis#[42] True False
            '''
            joint = self.pose_dataset.get_joint(idx)
            joint = torch.from_numpy(joint)
            '''
            if 'base_joints' in query:
                sample['joints']=joint
            '''
            K = self.pose_dataset.get_K(idx)
            K = torch.from_numpy(K)
            '''
            if 'base_Ks' in query:
                sample['Ks']=K
            '''
            depth = self.pose_dataset.get_depth(idx)#[320,320]
            '''
            if 'base_depth' in query:
                base_depth = np.array(depth)
                base_depth = depth_two_uint8_to_float(base_depth[:, :, 0], base_depth[:, :, 1])
                sample['depth']=func_transforms.to_tensor(base_depth).float()#[1,320,320]
            '''
            joints2d_l = joints2d[:21,:]
            joints2d_r = joints2d[21:,:]

            joint_l = joint[:21,:]
            joint_r = joint[21:,:]

            if hand_side:#1 right; 0 left
                xyz21 = joint_r
                uv21 = joints2d_r
                uv_vis = joints2d_vis_r
                mask_vis = mask_r
                mask_vis = Image.fromarray(np.array(mask_vis.float().squeeze(0)))
            else:
                # flip to right
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                xyz21 = joint_l
                xyz21[:, 0] = -joint_l[:, 0]
                uv21 = joints2d_l
                uv21[:,0] = self.inp_res - joints2d_l[:,0]
                uv_vis = joints2d_vis_l
                mask_vis = mask_l
                mask_vis = Image.fromarray(np.array(mask_vis.float().squeeze(0)))
                mask_vis = mask_vis.transpose(Image.FLIP_LEFT_RIGHT)
            #sample['mask_vis'] = mask_vis
            if 'joints' in query:
                sample['xyz21']=xyz21
            '''
            if "open_2dj" in query:
                # only for right hand
                open_2dj = self.pose_dataset.get_open_2dj(idx)
                # not true for left hand
                open_2dj_con = self.pose_dataset.get_open_2dj_con(idx)
                sample['open_2dj'] = open_2dj
                sample['open_2dj_con'] = open_2dj_con
            '''

            #if 'base_joints2d' in query:
            sample['uv21']=uv21
            sample['flip_sides']=torch.ones_like(hand_side).int()
            
            sample['uv_vis']=uv_vis
            
            # make coords relative to root joint
            joint_root = xyz21[0,:]
            joint_rel = xyz21 - joint_root
            index_root_bone_length = torch.sqrt(torch.sum((joint_rel[12, :] - joint_rel[11, :])**2))
            sample['keypoint_scale'] = index_root_bone_length
            sample['xyz21_normed'] = xyz21/index_root_bone_length

            """Hand CROP"""
            crop_center = uv21[12,:]#[2]
            #crop_center.view(2)
            sample['crop_center0']=crop_center

            #415
            '''
            # rotation
            # update image,mask_vis,depth,uv21,xyz21,K
            #import pdb;pdb.set_trace()
            if self.train:
                rot = np.random.uniform(low=-np.pi, high=np.pi)
                self.crop_center_noise = True
            else:
                rot = 0
                self.crop_center_noise = False
            rot_mat = np.array(
                [
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1],
                ]
            ).astype(np.float32)
            if self.train:
                xyz21 = rot_mat.dot(
                            xyz21.transpose(1, 0)
                        ).transpose()
            '''
            #415
            self.crop_center_noise = False
            if self.crop_center_noise:
                noise = 5 * torch.randn([2])#torch.normal(0, 20, size=(2))
                crop_center = noise + crop_center
            #sample['crop_center']=crop_center
            #sample['noise']=noise

            crop_scale_noise = torch.ones(1)

            self.crop_scale_noise = True
            if self.crop_scale_noise:
                crop_scale_noise = (1 - 1.1) * torch.rand(1) + 1 - 0.1#(1.2 - 1) * torch.rand(1) + 1
            #sample['crop_scale_noise']=crop_scale_noise
            
            # select visible coords only
            #uv_h = uv21[:,1]*((uv_vis==True).float())
            #uv_w = uv21[:,0]*((uv_vis==True).float())
            # select all coords
            uv_h = uv21[:,1]
            uv_w = uv21[:,0]
            #uv_hw = torch.stack([uv_h,uv_w],1)
            uv_hw = torch.stack([uv_w,uv_h],1)
            #sample['uv_hw']=uv_hw
            min_uv = torch.max(torch.min(uv_hw,0)[0],torch.zeros(2))
            max_uv = torch.min(torch.max(uv_hw,0)[0],torch.ones(2)*self.inp_res)
            #sample['max_uv']=max_uv
            #sample['min_uv']=min_uv
            #crop_size_best = 2*torch.max(max_uv-crop_center,crop_center-min_uv)
            crop_size_best = 3*torch.max(max_uv-crop_center,crop_center-min_uv)
            #sample['crop_size_best0']=crop_size_best

            crop_size_best = torch.max(crop_size_best)
            crop_size_best = torch.min(torch.max(crop_size_best,torch.ones(1)*50.0),torch.ones(1)*500.0)
            sample['crop_size_best']=crop_size_best

            # calculate necessary scaling
            scale = self.inp_res1 / crop_size_best
            scale = torch.min(torch.max(scale,torch.ones(1)),torch.ones(1)*10.0)
            scale = scale * crop_scale_noise
            sample['crop_scale'] = scale

            #415
            '''
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                crop_center, scale, [self.inp_res, self.inp_res], rot=rot
            )
            uv21 = handutils.transform_coords(uv21, affinetrans)
            uv21 = torch.from_numpy(uv21).float()
            sample['affinetrans']=affinetrans
            sample['post_rot_trans']=post_rot_trans
            sample['uv210']=uv21
            
            sample['Ks']=K
            K = post_rot_trans.dot(K)
            K = torch.from_numpy(K)
            
            sample['K']=K
            image = handutils.transform_img(
                        image, affinetrans, [self.inp_res, self.inp_res]
                    )
            depth = handutils.transform_img(
                        depth, affinetrans, [self.inp_res, self.inp_res]
                    )
            mask_vis = handutils.transform_img(
                        mask_vis, affinetrans, [self.inp_res, self.inp_res]
                    )
            #sample['image']=image

            crop_center = uv21[12,:]#[2]
            crop_size_scales = self.inp_res1
            y1 = crop_center[1] - crop_size_scales//2
            y2 = y1 + crop_size_scales
            #x1 = crop_center[1] - crop_size_scales//2
            x1 = crop_center[0] - crop_size_scales//2
            x2 = x1 + crop_size_scales
            sample['y1']=y1
            sample['y2']=y2
            sample['x1']=x1
            sample['x2']=x2

            img_crop = func_transforms.resized_crop(image, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            if 'trans_images' in query:
                sample['img_crop'] = func_transforms.to_tensor(img_crop).float()
            depth_crop = func_transforms.resized_crop(depth, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            depth_crop = np.array(depth_crop)
            depth_crop = depth_two_uint8_to_float(depth_crop[:, :, 0], depth_crop[:, :, 1])
            if 'trans_depth' in query:
                sample['depth_crop']=func_transforms.to_tensor(depth_crop).float()#[1,320,320]
            mask_crop = func_transforms.resized_crop(mask_vis, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            if 'trans_masks' in query:
                mask_crop = ((func_transforms.to_tensor(mask_crop)*255)>0).int()
                sample['mask_crop'] = mask_crop


            # Modify uv21
            #uv21_u = (uv21[:,0] - crop_center[1]) * scale + self.inp_res1 // 2
            #uv21_v = (uv21[:,1] - crop_center[0]) * scale + self.inp_res1 // 2
            scale = 1
            uv21_u = (uv21[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
            uv21_v = (uv21[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
            uv21_crop = torch.stack([uv21_u,uv21_v],1)
            if 'trans_joints2d' in query:
                sample['uv21_crop'] = uv21_crop
            # Modify camera intrinsics
            scale_matrix = torch.tensor([[scale,0.0,0.0],[0.0,scale,0.0],[0.0,0.0,1.0]])
            trans1 = crop_center[0] * scale - self.inp_res1 // 2
            trans2 = crop_center[1] * scale - self.inp_res1 // 2
            #trans_matrix = torch.tensor([[1.0,0.0,-trans2],[0.0,1.0,-trans1],[0.0,0.0,1.0]])
            trans_matrix = torch.tensor([[1.0,0.0,-trans1],[0.0,1.0,-trans2],[0.0,0.0,1.0]])
            #sample['scale_matrix']=scale_matrix
            #sample['trans_matrix']=trans_matrix
            K_crop = torch.mm(trans_matrix, torch.mm(scale_matrix, K))
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            '''
            #415 
            
            # Crop image
            crop_size_scales = self.inp_res1 / scale
            sample['crop_size_scales'] = crop_size_scales
            #y1 = crop_center[0] - crop_size_scales//2
            y1 = crop_center[1] - crop_size_scales//2
            y2 = y1 + crop_size_scales
            #x1 = crop_center[1] - crop_size_scales//2
            x1 = crop_center[0] - crop_size_scales//2
            x2 = x1 + crop_size_scales
            sample['y1']=y1
            sample['y2']=y2
            sample['x1']=x1
            sample['x2']=x2

            img_crop = func_transforms.resized_crop(image, y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
            if 'trans_images' in query:
                sample['img_crop'] = func_transforms.to_tensor(img_crop).float()
            depth_crop = func_transforms.resized_crop(depth, y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
            depth_crop = np.array(depth_crop)
            depth_crop = depth_two_uint8_to_float(depth_crop[:, :, 0], depth_crop[:, :, 1])
            if 'trans_depth' in query:
                sample['depth_crop']=func_transforms.to_tensor(depth_crop).float()#[1,320,320]
            mask_crop = func_transforms.resized_crop(mask_vis, y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
            if 'trans_masks' in query:
                mask_crop = ((func_transforms.to_tensor(mask_crop)*255)>0).int()
                sample['mask_crop'] = mask_crop


            # Modify uv21
            #uv21_u = (uv21[:,0] - crop_center[1]) * scale + self.inp_res1 // 2
            #uv21_v = (uv21[:,1] - crop_center[0]) * scale + self.inp_res1 // 2
            
            uv21_u = (uv21[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
            uv21_v = (uv21[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
            uv21_crop = torch.stack([uv21_u,uv21_v],1)
            if 'trans_joints2d' in query:
                sample['uv21_crop'] = uv21_crop
            '''
            if "open_2dj" in query:
                open_2dj_u = (open_2dj[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
                open_2dj_v = (open_2dj[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
                open_2dj_crop = torch.stack([open_2dj_u,open_2dj_v],1)
                sample['open_2dj_crop'] = open_2dj_crop
            '''

            # Modify camera intrinsics
            scale_matrix = torch.tensor([[scale,0.0,0.0],[0.0,scale,0.0],[0.0,0.0,1.0]])
            trans1 = crop_center[0] * scale - self.inp_res1 // 2
            trans2 = crop_center[1] * scale - self.inp_res1 // 2
            #trans_matrix = torch.tensor([[1.0,0.0,-trans2],[0.0,1.0,-trans1],[0.0,0.0,1.0]])
            trans_matrix = torch.tensor([[1.0,0.0,-trans1],[0.0,1.0,-trans2],[0.0,0.0,1.0]])
            #sample['scale_matrix']=scale_matrix
            #sample['trans_matrix']=trans_matrix
            K_crop = torch.mm(trans_matrix, torch.mm(scale_matrix, K))
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            
            # translate camera and 3d joints, keep 2d
            '''
            K_crop_trans = K_crop.clone()
            K_crop_trans[0,2], K_crop_trans[1,2] = self.inp_res1 // 2, self.inp_res1 // 2
            sample['K_crop_trans'] = K_crop_trans

            #xyz21#[21,3]
            xyz_trans_matrix = torch.tensor([[1.0,0.0,K_crop_trans[0,2]- K_crop[0,2]],[0.0,1.0,K_crop_trans[1,2]- K_crop[1,2]],[0.0,0.0,1.0]])
            sample['xyz_trans_matrix'] = xyz_trans_matrix
            xyz_trans = torch.mm(xyz21,xyz_trans_matrix)
            sample['xyz_trans']=xyz_trans
            sample['Ks']=K
            '''
            # Data augmentation
            '''
            if self.train and needs_center_scale:
                # Randomly jitter center
                # Center is located in square of size 2*center_jitter_factor
                # in center of cropped image
                center_offsets = (
                    self.center_jittering
                    * scale
                    * np.random.uniform(low=-1, high=1, size=2)
                )
                center = center + center_offsets.astype(int)

                # Scale jittering
                scale_jittering = self.scale_jittering * np.random.randn() + 1
                scale_jittering = np.clip(
                    scale_jittering,
                    1 - self.scale_jittering,
                    1 + self.scale_jittering,
                )
                scale = scale * scale_jittering

                rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
            else:
                rot = 0
            '''

            '''
            rot_mat = np.array(
                [
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1],
                ]
            ).astype(np.float32)
            '''

            '''
            # Get 2D hand joints
            if ('trans_joints2d' in query) or ('trans_images' in query):
                affinetrans, post_rot_trans = handutils.get_affine_transform(
                    center, scale, [self.inp_res, self.inp_res], rot=rot
                )
            if ('base_joints2d' in query) or ('trans_joints2d' in query):
                joints2d = self.pose_dataset.get_joints2d(idx)
                if 'base_joints2d' in query:
                    sample['joints2d'] = torch.from_numpy(joints2d)
                if 'trans_joints2d' in query:
                    rows = handutils.transform_coords(joints2d, affinetrans)
                    sample['trans_joints2d'] = torch.from_numpy(np.array(rows))
            
            # Get camintrs
            if 'base_Ks' in query or ('trans_Ks' in query):
                K = self.pose_dataset.get_K(idx)
                if 'base_Ks' in query:
                    sample['Ks']=torch.from_numpy(K)
                if 'trans_Ks' in query:
                    # Rotation is applied as extr transform
                    new_camintr = post_rot_trans.dot(K)
                    sample['trans_Ks'] = torch.from_numpy(new_camintr)
            
            # Get 3d joints
            if 'base_joints' in query or ('trans_joints' in query):
                joint = self.pose_dataset.get_joint(idx)
                if 'base_joints' in query:
                    sample['joints']=torch.from_numpy(joint)
                
                if self.train:
                    joints = rot_mat.dot(
                        joints.transpose(1, 0)
                    ).transpose()
                
                if 'trans_joints' in query:
                    sample['trans_joints'] = torch.from_numpy(joint)
            
            # Get segmentation
            if 'base_masks' in query or ('trans_masks' in query):
                mask = self.pose_dataset.get_mask(idx)#[320,320]
                if 'base_masks' in query:
                    sample['masks']=(func_transforms.to_tensor(mask)*255).int()
                if 'trans_masks' in query:
                    mask = handutils.transform_img(
                        mask, affinetrans, [self.inp_res, self.inp_res]
                    )
                    mask = mask.crop((0, 0, self.inp_res1, self.inp_res1))
                    mask = (func_transforms.to_tensor(mask)*255).int()
                    sample['trans_masks'] = mask
            
            # Get depth
            if 'base_depth' in query or ('trans_depth' in query):
                depth = self.pose_dataset.get_depth(idx)#[320,320]
                if 'base_depth' in query:
                    base_depth = np.array(depth)
                    base_depth = depth_two_uint8_to_float(base_depth[:, :, 0], base_depth[:, :, 1])
                    sample['depth']=func_transforms.to_tensor(base_depth).float()
                if 'trans_depth' in query:
                    depth = handutils.transform_img(
                        depth, affinetrans, [self.inp_res, self.inp_res]
                    )
                    depth = depth.crop((0, 0, self.inp_res1, self.inp_res1))
                    trans_depth = np.array(depth)
                    trans_depth = depth_two_uint8_to_float(trans_depth[:, :, 0], trans_depth[:, :, 1])

                    sample['trans_depth'] = func_transforms.to_tensor(trans_depth).float()
                    

            # Get trans image
            if 'trans_images' in query:
                 # Transform and crop
                image = handutils.transform_img(
                    image, affinetrans, [self.inp_res, self.inp_res]
                )
                image = image.crop((0, 0, self.inp_res1, self.inp_res1))
                image = func_transforms.to_tensor(image).float()
                if self.black_padding:
                    padding_ratio = 0.2
                    padding_size = int(self.inp_res * padding_ratio)
                    image[:, 0:padding_size, :] = 0
                    image[:, -padding_size:-1, :] = 0
                    image[:, :, 0:padding_size] = 0
                    image[:, :, -padding_size:-1] = 0
                sample['trans_images'] = image
            '''

            '''
            if 'masks' in query:
                mask = self.pose_dataset.get_mask(idx)
                sample['masks']=mask
            if 'Ks' in query:
                K = self.pose_dataset.get_K(idx)
                sample['Ks']=K
            if 'joints' in query:
                joint = self.pose_dataset.get_joint(idx)
                sample['joints']=joint
            if 'joints2d' in query:
                joints2d = self.pose_dataset.get_joints2d(idx)
                sample['joints2d']=joints2d
            if 'depth' in query:
                depth = self.pose_dataset.get_depth(idx)
                sample['depth']=depth
            '''
        # RHD
        if self.dat_name == 'RHD0':
            #print("I am here!")
            # Get original image
            '''
            if 'base_images' in query or 'trans_images' in query:
                center, scale = self.pose_dataset.get_center_scale(idx)
                needs_center_scale = True
                image = self.pose_dataset.get_img(idx)
                if 'base_images' in query:
                    sample['images']=func_transforms.to_tensor(image).float()
            else:
                needs_center_scale = False
            '''
            image = self.pose_dataset.get_img(idx)
            
            if 'base_images' in query:
                sample['images']=func_transforms.to_tensor(image).float()
            
            mask_ori = self.pose_dataset.get_mask(idx)#[320,320]
            mask = (func_transforms.to_tensor(mask_ori)*255).int()
            
            if 'base_masks' in query:
                sample['masks']=mask
            
            #needs_center_scale=False
            mask_r = (mask>17)
            mask_l = (mask>1) - mask_r
            #one_map, zero_map = torch.ones_like(mask), torch.zeros_like(mask)
            num_px_left_hand = torch.sum(mask_l)
            num_px_right_hand = torch.sum(mask_r)

            sample['mask_r']=mask_r
            sample['mask_l']=mask_l
            sample['num_l']=num_px_left_hand
            sample['num_r']=num_px_right_hand
            hand_side = torch.where(num_px_left_hand>num_px_right_hand,torch.zeros(1).int(),torch.ones(1).int())
            # 0 left; 1 right
            if 'sides' in query:
                sample['sides'] = hand_side
            
            joints2d = self.pose_dataset.get_joints2d(idx)
            joints2d = torch.from_numpy(joints2d)
            '''
            if 'base_joints2d' in query:
                sample['joints2d'] = joints2d
            '''
            joints2d_vis = self.pose_dataset.get_j2dvis(idx)
            '''
            if 'joints2d_vis' in query:
                sample['joints2d_vis']=joints2d_vis#[42] True False
            '''
            joint = self.pose_dataset.get_joint(idx)
            joint = torch.from_numpy(joint)
            '''
            if 'base_joints' in query:
                sample['joints']=joint
            '''
            K = self.pose_dataset.get_K(idx)
            K = torch.from_numpy(K)
            '''
            if 'base_Ks' in query:
                sample['Ks']=K
            '''
            depth = self.pose_dataset.get_depth(idx)#[320,320]
            '''
            if 'base_depth' in query:
                base_depth = np.array(depth)
                base_depth = depth_two_uint8_to_float(base_depth[:, :, 0], base_depth[:, :, 1])
                sample['depth']=func_transforms.to_tensor(base_depth).float()#[1,320,320]
            '''
            joints2d_l = joints2d[:21,:]
            joints2d_r = joints2d[21:,:]
            #import pdb;pdb.set_trace()
            
            joints2d_vis_l = joints2d_vis[:21]
            joints2d_vis_r = joints2d_vis[21:]

            joint_l = joint[:21,:]
            joint_r = joint[21:,:]

            if hand_side:#1 right; 0 left
                xyz21 = joint_r
                uv21 = joints2d_r
                uv_vis = joints2d_vis_r
                mask_vis = mask_r
                mask_vis = Image.fromarray(np.array(mask_vis.float().squeeze(0)))
            else:
                # flip to right
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                xyz21 = joint_l
                xyz21[:, 0] = -joint_l[:, 0]
                uv21 = joints2d_l
                uv21[:,0] = self.inp_res - joints2d_l[:,0]
                uv_vis = joints2d_vis_l
                mask_vis = mask_l
                mask_vis = Image.fromarray(np.array(mask_vis.float().squeeze(0)))
                mask_vis = mask_vis.transpose(Image.FLIP_LEFT_RIGHT)
            #sample['mask_vis'] = mask_vis
            if 'joints' in query:
                sample['xyz21']=xyz21
            #if 'base_joints2d' in query:
            sample['uv21']=uv21
            sample['flip_sides']=torch.ones_like(hand_side).int()
            
            sample['uv_vis']=uv_vis
            
            # make coords relative to root joint
            joint_root = xyz21[0,:]
            joint_rel = xyz21 - joint_root
            index_root_bone_length = torch.sqrt(torch.sum((joint_rel[12, :] - joint_rel[11, :])**2))
            sample['keypoint_scale'] = index_root_bone_length
            sample['xyz21_normed'] = xyz21/index_root_bone_length

            #joints2d = self.get_joints2d(idx)
            center = handutils.get_annot_center(np.array(uv21))
            scale = handutils.get_annot_scale(
                np.array(uv21), scale_factor=2.5
            )
            if self.train:
                center_offsets = (
                    self.center_jittering
                    * scale
                    * np.random.uniform(low=-1, high=1, size=2)
                )
                center = center + center_offsets.astype(int)

                # Scale jittering
                scale_jittering = self.scale_jittering * np.random.randn() + 1
                scale_jittering = np.clip(
                    scale_jittering,
                    1 - self.scale_jittering,
                    1 + self.scale_jittering,
                )
                scale = scale * scale_jittering

                rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
            else:
                rot = 0
            rot_mat = np.array(
                [
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1],
                ]
            ).astype(np.float32)

            #415
            
            # rotation
            # update image,mask_vis,depth,uv21,xyz21,K
            #import pdb;pdb.set_trace()
            
            xyz21 = rot_mat.dot(
                        xyz21.transpose(1, 0)
                    ).transpose()
            

            #415
            
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                center, scale, [self.inp_res, self.inp_res], rot=rot
            )
            uv21 = handutils.transform_coords(uv21, affinetrans)
            uv21 = torch.from_numpy(uv21).float()
            sample['affinetrans']=affinetrans
            sample['post_rot_trans']=post_rot_trans
            #sample['uv210']=uv21
            
            sample['Ks']=K
            K = post_rot_trans.dot(K)
            K = torch.from_numpy(K)
            
            sample['K']=K
            image = handutils.transform_img(
                        image, affinetrans, [self.inp_res, self.inp_res]
                    )
            depth = handutils.transform_img(
                        depth, affinetrans, [self.inp_res, self.inp_res]
                    )
            mask_vis = handutils.transform_img(
                        mask_vis, affinetrans, [self.inp_res, self.inp_res]
                    )
            #sample['image']=image
            sample['img_crop'] = func_transforms.to_tensor(image).float()
            sample['depth_crop'] = func_transforms.to_tensor(depth).float()
            sample['mask_crop'] = func_transforms.to_tensor(mask_vis).float()
            sample['uv21_crop'] = uv21
            sample['K_crop'] = K
            '''
            crop_center = uv21[12,:]#[2]
            crop_size_scales = self.inp_res1
            y1 = crop_center[1] - crop_size_scales//2
            y2 = y1 + crop_size_scales
            #x1 = crop_center[1] - crop_size_scales//2
            x1 = crop_center[0] - crop_size_scales//2
            x2 = x1 + crop_size_scales
            sample['y1']=y1
            sample['y2']=y2
            sample['x1']=x1
            sample['x2']=x2
            

            img_crop = func_transforms.resized_crop(image, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            if 'trans_images' in query:
                sample['img_crop'] = func_transforms.to_tensor(img_crop).float()
            depth_crop = func_transforms.resized_crop(depth, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            depth_crop = np.array(depth_crop)
            depth_crop = depth_two_uint8_to_float(depth_crop[:, :, 0], depth_crop[:, :, 1])
            if 'trans_depth' in query:
                sample['depth_crop']=func_transforms.to_tensor(depth_crop).float()#[1,320,320]
            mask_crop = func_transforms.resized_crop(mask_vis, y1.data.item(), x1.data.item(), crop_size_scales, crop_size_scales, [self.inp_res1,self.inp_res1])
            if 'trans_masks' in query:
                mask_crop = ((func_transforms.to_tensor(mask_crop)*255)>0).int()
                sample['mask_crop'] = mask_crop
            

            # Modify uv21
            #uv21_u = (uv21[:,0] - crop_center[1]) * scale + self.inp_res1 // 2
            #uv21_v = (uv21[:,1] - crop_center[0]) * scale + self.inp_res1 // 2
            scale = 1
            uv21_u = (uv21[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
            uv21_v = (uv21[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
            uv21_crop = torch.stack([uv21_u,uv21_v],1)
            if 'trans_joints2d' in query:
                sample['uv21_crop'] = uv21_crop
            # Modify camera intrinsics
            scale_matrix = torch.tensor([[scale,0.0,0.0],[0.0,scale,0.0],[0.0,0.0,1.0]])
            trans1 = crop_center[0] * scale - self.inp_res1 // 2
            trans2 = crop_center[1] * scale - self.inp_res1 // 2
            #trans_matrix = torch.tensor([[1.0,0.0,-trans2],[0.0,1.0,-trans1],[0.0,0.0,1.0]])
            trans_matrix = torch.tensor([[1.0,0.0,-trans1],[0.0,1.0,-trans2],[0.0,0.0,1.0]])
            #sample['scale_matrix']=scale_matrix
            #sample['trans_matrix']=trans_matrix
            K_crop = torch.mm(trans_matrix, torch.mm(scale_matrix, K))
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            '''
            #415 
        if self.dat_name == 'HO3D':
            image = self.pose_dataset.get_img(idx)#[3,480,640]
            if 'base_images' in query:
                sample['base_images']=func_transforms.to_tensor(image).float()
            
            if 'trans_masks' in query or 'base_masks' in query:
                mask = self.pose_dataset.get_masks(idx)
                base_masks = torch.round(func_transforms.to_tensor(mask))
                if 'base_masks' in query:
                    sample['base_masks']=base_masks
                hand_mask = base_masks[0]
                obj_mask = base_masks[2]
            
            if 'base_depths' in query:
                depth = self.pose_dataset.get_depth(idx)#[3,480,640]
                '''
                base_depth = func_transforms.to_tensor(depth)
                depth_scale = 0.00012498664727900177
                dpt = base_depth[:, :, 2] + base_depth[:, :, 1] * 256
                dpt = dpt * depth_scale
                sample['base_depths']=dpt
                '''
                base_depth = np.array(depth)
                base_depth = depth_two_uint8_to_float(base_depth[:, :, 0], base_depth[:, :, 1])
                sample['depth']=func_transforms.to_tensor(base_depth).float()#[1,480,640]

            if 'open_2dj' in query or "trans_open_2dj" in query:
                open_2dj = self.pose_dataset.get_open_2dj(idx)
                sample['open_2dj']=open_2dj
                open_2dj_con = self.pose_dataset.get_open_2dj_con(idx)
                sample['open_2dj_con']=open_2dj_con

            meta = self.pose_dataset.get_meta(idx)
                        
            if 'handPose' in meta.keys():
                sample['hand_pose'] = torch.from_numpy(meta['handPose']).float()
            if 'handBeta' in meta.keys():
                sample['hand_shape'] = torch.from_numpy(meta['handBeta']).float()
            if 'camMat' in meta.keys():
                Ks = torch.from_numpy(meta['camMat']).float()
                Ks = torch.mm(Ks,self.pose_dataset.cam_extr)#check merge cam extr to cam intr
                sample['camMat'] = Ks
            if 'handTrans' in meta.keys():
                sample['handTrans'] = torch.from_numpy(meta['handTrans']).float()
            if 'handJoints3D' in meta.keys():
                j3d = torch.from_numpy(meta['handJoints3D']).float()
                #j3d[:,0]=-j3d[:,0]
                
            if 'objCorners3D' in meta.keys():
                obj6d = torch.from_numpy(meta['objCorners3D']).float()
                uv6 = proj_func(obj6d.unsqueeze(0), Ks.unsqueeze(0))
                # side = 'right'
                uv6 = uv6.squeeze(0)
                sample['uv6'] = uv6
            
            if 'handBoundingBox' in meta.keys():#for evaluation set
                uv21 = torch.tensor([[meta['handBoundingBox'][0],meta['handBoundingBox'][1]],[meta['handBoundingBox'][2],meta['handBoundingBox'][3]]])
                
                # add
                sample['root_xyz'] = j3d
                
                if 'xyz21_normed' in query:
                    joint_root = j3d[0,:]
                    joint_rel = j3d - joint_root
                    index_root_bone_length = torch.sqrt(torch.sum((joint_rel[5, :] - joint_rel[4, :])**2))
                    sample['keypoint_scale'] = index_root_bone_length
                    sample['xyz21_normed'] = j3d/index_root_bone_length
            else:
                uv21 = proj_func(j3d.unsqueeze(0), Ks.unsqueeze(0))
                # make coords relative to root joint
                sample['xyz21'] = j3d
                if 'xyz21_normed' in query:
                    joint_root = j3d[0,:]
                    joint_rel = j3d - joint_root
                    index_root_bone_length = torch.sqrt(torch.sum((joint_rel[5, :] - joint_rel[4, :])**2))
                    sample['keypoint_scale'] = index_root_bone_length
                    sample['xyz21_normed'] = j3d/index_root_bone_length
            uv21 = uv21.squeeze(0)
            sample['uv21'] = uv21

            """Hand CROP"""
            ho_scope = 0
            if ho_scope:
                uv_all = torch.cat((uv21,uv6),0)
                scale_num = 2
            else:
                uv_all = uv21
                scale_num = 4
            #crop_center = torch.mean(uv_all,0)
            crop_center = (torch.max(uv_all,0)[0]+torch.min(uv_all,0)[0])/2
            #crop_center = uv21[4,:]#[2]
            #crop_center.view(2)
            sample['crop_center0']=crop_center
            self.crop_center_noise = True
            if self.crop_center_noise:
                noise = 5 * torch.randn([2])#torch.normal(0, 20, size=(2))
                crop_center = noise + crop_center
            
            crop_scale_noise = torch.ones(1)
            self.crop_scale_noise = True
            if self.crop_scale_noise:
                crop_scale_noise = (1 - 1.1) * torch.rand(1) + 1 - 0.1#(1.2 - 1) * torch.rand(1) + 1
            #sample['crop_scale_noise']=crop_scale_noise
            #uv_h = uv21[:,1]
            #uv_w = uv21[:,0]
            uv_h = uv_all[:,1]
            uv_w = uv_all[:,0]
            uv_hw = torch.stack([uv_w,uv_h],1)
            min_uv = torch.max(torch.min(uv_hw,0)[0],torch.zeros(2)) - torch.tensor([10.0,10.0])
            #max_uv = torch.min(torch.max(uv_hw,0)[0],torch.ones(2)*self.inp_res)
            max_uv = torch.min(torch.max(uv_hw,0)[0],torch.tensor([640.0,480.0])) + torch.tensor([10.0,10.0])
            #sample['max_uv'] = max_uv
            #sample['min_uv'] = min_uv
            #sample['uv_hw'] = uv_hw

            crop_size_best = scale_num*torch.max(max_uv-crop_center,crop_center-min_uv)# 2*
            crop_size_best = torch.max(crop_size_best)
            crop_size_best = torch.min(torch.max(crop_size_best,torch.ones(1)*50.0),torch.ones(1)*640.0)
            #sample['crop_size_best']=crop_size_best
            # calculate necessary scaling
            scale = self.inp_res1 / crop_size_best
            #scale = torch.min(torch.max(scale,torch.ones(1)),torch.ones(1)*10.0)#check
            scale = torch.min(scale,torch.ones(1)*10.0)
            scale = scale * crop_scale_noise
            sample['crop_scale'] = scale
            # Crop image
            crop_size_scales = self.inp_res1 / scale
            sample['crop_size_scales'] = crop_size_scales
            #y1 = crop_center[0] - crop_size_scales//2
            y1 = crop_center[1] - crop_size_scales//2
            y2 = y1 + crop_size_scales
            #x1 = crop_center[1] - crop_size_scales//2
            x1 = crop_center[0] - crop_size_scales//2
            x2 = x1 + crop_size_scales
            sample['y1']=y1
            sample['y2']=y2
            sample['x1']=x1
            sample['x2']=x2

            img_crop = func_transforms.resized_crop(image, y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
            if 'trans_images' in query:
                sample['img_crop'] = func_transforms.to_tensor(img_crop).float()
            
            if 'trans_depth' in query:
                depth_crop = func_transforms.resized_crop(depth, y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
                depth_crop = np.array(depth_crop)
                depth_crop = depth_two_uint8_to_float(depth_crop[:, :, 0], depth_crop[:, :, 1])
                sample['depth_crop']=func_transforms.to_tensor(depth_crop).float()#[1,320,320]
            
            if 'trans_masks' in query or 'base_masks' in query:
                hand_mask_crop = func_transforms.resized_crop(mask.getchannel(0), y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1],interpolation=1)
                if 'trans_masks' in query:
                    #hand_mask_crop = ((func_transforms.to_tensor(hand_mask_crop)*255)>0).int()
                    hand_mask_crop = (func_transforms.to_tensor(hand_mask_crop)).round()
                    sample['hand_mask_crop'] = hand_mask_crop
                obj_mask_crop = func_transforms.resized_crop(mask.getchannel(2), y1.data.item(), x1.data.item(), crop_size_scales.data.item(), crop_size_scales.data.item(), [self.inp_res1,self.inp_res1])
                if 'trans_masks' in query:
                    obj_mask_crop = (func_transforms.to_tensor(obj_mask_crop)).round()
                    sample['obj_mask_crop'] = obj_mask_crop

            # Modify uv21
            uv21_u = (uv21[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
            uv21_v = (uv21[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
            uv21_crop = torch.stack([uv21_u,uv21_v],1)
            if 'trans_joints2d' in query:
                sample['uv21_crop'] = uv21_crop

            # Modify uv6
            if 'uv6_crop' in query:
                uv6_u = (uv6[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
                uv6_v = (uv6[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
                uv6_crop = torch.stack([uv6_u,uv6_v],1)
                sample['uv6_crop'] = uv6_crop
            # Modify openpose
            if 'open_2dj' in query or "trans_open_2dj" in query:
                open_2dj_u = (open_2dj[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
                open_2dj_v = (open_2dj[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
                open_2dj_crop = torch.stack([open_2dj_u,open_2dj_v],1)
                sample['open_2dj_crop']=open_2dj_crop

            # Modify camera intrinsics
            scale_matrix = torch.tensor([[scale,0.0,0.0],[0.0,scale,0.0],[0.0,0.0,1.0]])
            trans1 = crop_center[0] * scale - self.inp_res1 // 2
            trans2 = crop_center[1] * scale - self.inp_res1 // 2
            trans_matrix = torch.tensor([[1.0,0.0,-trans1],[0.0,1.0,-trans2],[0.0,0.0,1.0]])
            K_crop = torch.mm(trans_matrix, torch.mm(scale_matrix, Ks))
            #
            sample['scale_matrix'] = scale_matrix
            sample['trans_matrix'] = trans_matrix
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop

        if self.dat_name == 'HO3D0':
            # raw image 
            image = self.pose_dataset.get_img(idx)#[3,480,640]
            sample['images']=func_transforms.to_tensor(image).float()#image
            if 'open_2dj' in query or "trans_open_2dj" in query:
                open_2dj = self.pose_dataset.get_open_2dj(idx)
                sample['open_2dj']=open_2dj
                open_2dj_con = self.pose_dataset.get_open_2dj_con(idx)
                sample['open_2dj_con']=open_2dj_con
            meta = self.pose_dataset.get_meta(idx)
            if 'camMat' in meta.keys():
                Ks = torch.from_numpy(meta['camMat']).float()
                Ks = torch.mm(Ks,self.pose_dataset.cam_extr)#check merge cam extr to cam intr
                sample['camMat'] = Ks
            
            if 'handBoundingBox' in meta.keys():#for evaluation set
                uv21 = torch.tensor([[meta['handBoundingBox'][0],meta['handBoundingBox'][1]],[meta['handBoundingBox'][2],meta['handBoundingBox'][3]]])
            else:
                if 'handJoints3D' in meta.keys():
                    j3d = torch.from_numpy(meta['handJoints3D']).float()
                    uv21 = proj_func(j3d.unsqueeze(0), Ks.unsqueeze(0))
                    sample['j3d_raw'] = j3d
                # make coords relative to root joint
                
            uv21 = uv21.squeeze(0)
            sample['uv21'] = uv21
            
            #resize_img = func_transforms.resize(resize_image,[224,224])
            #resize_img = nn.functional.interpolate(resize_image,size=(224,224))

            uv_h = uv21[:,1]
            uv_w = uv21[:,0]
            uv_hw = torch.stack([uv_w,uv_h],1)
            min_uv = torch.max(torch.min(uv_hw,0)[0],torch.zeros(2)) - torch.tensor([40.0,40.0])
            #max_uv = torch.min(torch.max(uv_hw,0)[0],torch.ones(2)*self.inp_res)
            max_uv = torch.min(torch.max(uv_hw,0)[0],torch.tensor([640.0,480.0])) + torch.tensor([40.0,40.0])
            sample['min_uv'] = min_uv
            sample['max_uv'] = max_uv
            crop_center = torch.tensor([320.0,240.0])
            crop_size_best = torch.max(max_uv-crop_center,crop_center-min_uv)
            crop_size_best = torch.max(crop_size_best)*2
            crop_size_best = torch.max(crop_size_best,torch.tensor(240.0))
            sample['crop_size_best'] = crop_size_best
            scale = self.inp_res1 / crop_size_best

            y1 = crop_center[1] - crop_size_best//2
            #y2 = y1 + crop_size_best
            #x1 = crop_center[1] - crop_size_scales//2
            x1 = crop_center[0] - crop_size_best//2
            #x2 = x1 + crop_size_best
            if self.train:
                rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
            else:
                rot = 0
            rot_mat = np.array(
                [
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1],
                ]
            ).astype(np.float32)
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                np.asarray([112, 112]), 224, [224, 224], rot=rot
            )
            sample['rot'] = rot
            img_crop = func_transforms.resized_crop(image, y1.data.item(), x1.data.item(), crop_size_best.data.item(), crop_size_best.data.item(), [self.inp_res1,self.inp_res1])
            # rotation
            img_crop = handutils.transform_img(
                img_crop, affinetrans, [224, 224]
            )
            if 'trans_images' in query:
                sample['img_crop'] = func_transforms.to_tensor(img_crop).float()
            
            # Modify uv21
            uv21_u = (uv21[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
            uv21_v = (uv21[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
            uv21_crop = torch.stack([uv21_u,uv21_v],1)
            # rotation
            uv21_crop = handutils.transform_coords(uv21_crop.numpy(),affinetrans)
            if 'trans_joints2d' in query:
                sample['uv21_crop'] = uv21_crop

            if 'open_2dj' in query or "trans_open_2dj" in query:
                open_2dj_u = (open_2dj[:,0] - crop_center[0]) * scale + self.inp_res1 // 2
                open_2dj_v = (open_2dj[:,1] - crop_center[1]) * scale + self.inp_res1 // 2
                open_2dj_crop = torch.stack([open_2dj_u,open_2dj_v],1)
                # rotation
                open_2dj_crop = handutils.transform_coords(open_2dj_crop.numpy(),affinetrans)
                sample['open_2dj_crop']=open_2dj_crop
            
            # Modify xyz
            if 'handJoints3D' in meta.keys() and "joints" in query:
                j3d = meta['handJoints3D']

                '''
                trans_j3d = rot_mat.dot(
                    j3d.transpose(1, 0)
                ).transpose()
                sample['xyz21'] = torch.from_numpy(trans_j3d).float()
                '''
                # for ho3d since coordinate changed
                rot0 = -rot
                rot_mat1 = np.array(
                    [
                        [np.cos(rot0), -np.sin(rot0), 0],
                        [np.sin(rot0), np.cos(rot0), 0],
                        [0, 0, 1],
                    ]
                ).astype(np.float32)

                trans_j3d0 = rot_mat1.dot(
                    j3d.transpose(1, 0)
                ).transpose()
                #sample['rot_mat1'] = rot_mat1
                sample['xyz21'] = torch.from_numpy(trans_j3d0).float()

            # Modify camera intrinsics
            matrix = torch.tensor([[scale,1.0,self.inp_res1/640],[1.0,scale,self.inp_res1/480],[1.0,1.0,1.0]])
            #matrix = torch.tensor([[scale,1.0,0.0],[1.0,scale,0.0],[1.0,1.0,1.0]])
            #trans_matrix = torch.tensor([[0.0,0.0,-self.inp_res1/2],[0.0,0.0,-self.inp_res1/2],[0.0,0.0,0.0]])
            K_crop = torch.mul(matrix, Ks)#+trans_matrix
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            '''
            scale_matrix = torch.tensor([[scale,0.0,0.0],[0.0,scale,0.0],[0.0,0.0,1.0]])
            trans1 = crop_center[0] * scale - self.inp_res1 // 2
            trans2 = crop_center[1] * scale - self.inp_res1 // 2
            trans_matrix = torch.tensor([[1.0,0.0,-trans1],[0.0,1.0,-trans2],[0.0,0.0,1.0]])
            K_crop = torch.mm(trans_matrix, torch.mm(scale_matrix, Ks))
            #
            sample['scale_matrix'] = scale_matrix
            sample['trans_matrix'] = trans_matrix
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            '''
            
            '''
            # Only padding and resize
            # image resize
            m = nn.ZeroPad2d((0,0,80,80))
            resize_image = m(sample['images'])
            if 'trans_images' in query:
                sample['img_crop'] = resize_image#func_transforms.to_tensor(img_crop).float()
            scale = self.inp_res1 / 640
            
            if 'trans_joints2d' in query:
                uv21_u = uv21[:,0] * scale
                uv21_v = (uv21[:,1] + 80) * scale
                uv21_crop = torch.stack([uv21_u,uv21_v],1)
                sample['uv21_crop'] = uv21_crop
            if 'open_2dj' in query:
                open_2dj_u = (open_2dj[:,0]) * scale
                open_2dj_v = (open_2dj[:,1] + 80) * scale
                open_2dj_crop = torch.stack([open_2dj_u,open_2dj_v],1)
                sample['open_2dj_crop']=open_2dj_crop
            # Modify camera intrinsics
            matrix = torch.tensor([[scale,1.0,scale],[1.0,scale,self.inp_res1/480],[1.0,1.0,1.0]])
            K_crop = torch.mul(matrix, Ks)
            if 'trans_Ks' in query:
                sample['K_crop'] = K_crop
            '''
    
        # Dart
        if self.dat_name == 'Dart':          
            dart_sample = self.pose_dataset.__getitem__(idx)
            
            for key in dart_sample:
                if key == 'image' or key == 'image_mask':
                    sample[key] = func_transforms.to_tensor(copy.deepcopy(dart_sample[key])).float()
                else:
                    sample[key] = copy.deepcopy(dart_sample[key])
            
        return sample
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            random_idx = random.randint(0, len(self))
            print("Encountered error processing sample {}, try to use a random idx {} instead".format(idx, random_idx))
            sample = self.get_sample(random_idx, self.queries)
        return sample


class FreiHand:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.load_dataset()
        self.name = "FreiHand"
        self.split = split

    # Annotations
    def load_dataset(self):
        if self.set_name == 'evaluation':
            dataset_name = 'evaluation'
        else:
            dataset_name = 'training'
        self.K_list = json_load(os.path.join(self.base_path, '%s_K.json' % dataset_name))
        self.scale_list = json_load(os.path.join(self.base_path, '%s_scale.json' % dataset_name))
        self.mano_list = json_load(os.path.join(self.base_path, '%s_mano.json' % dataset_name))
        self.joint_list = json_load(os.path.join(self.base_path, '%s_xyz.json' % dataset_name))
        self.verts_list = json_load(os.path.join(self.base_path, '%s_verts.json' % self.set_name))
        
        openpose_v2_path = '/home/zhuoran/HandRecon/mydata/'
                
        if self.set_name == 'training' or self.set_name == 'trainval_train' or self.set_name == 'trainval_val':# only 32560
            #self.open_2dj_lists = json_load('/data/FreiHand_save/debug/detect_all.json')
            # self.open_2dj_lists = json_load(os.path.join(self.base_path, 'openpose_v2/training', 'detect_all.json'))
            self.open_2dj_lists = json_load(os.path.join(openpose_v2_path, 'openpose_v2/training', 'detect_all.json'))
            # self.open_2dj_lists = json_load(os.path.join(self.base_path, 'outputs', 'freihand-train_openpose_keypoints.json'))
            self.open_2dj_list = self.open_2dj_lists[0]
            self.open_2dj_con_list = self.open_2dj_lists[1]
            #self.CRFmask_dir = '/data/FreiHand_save/CRFmask/training'
            self.CRFmask_dir = os.path.join(self.base_path, 'CRFmask/training')
            
            if self.set_name == 'trainval_train':
                self.K_list = self.K_list[:30000]
                self.scale_list = self.scale_list[:30000]
                self.mano_list = self.mano_list[:30000]
                self.joint_list = self.joint_list[:30000]
                self.verts_list = self.verts_list[:30000]
                self.open_2dj_list = self.open_2dj_list[:30000]
                self.open_2dj_con_list = self.open_2dj_con_list[:30000]
            elif self.set_name == 'trainval_val':
                self.K_list = self.K_list[30000:]
                self.scale_list = self.scale_list[30000:]
                self.mano_list = self.mano_list[30000:]
                self.joint_list = self.joint_list[30000:]
                self.verts_list = self.verts_list[30000:]
                self.open_2dj_list = self.open_2dj_list[30000:]
                self.open_2dj_con_list = self.open_2dj_con_list[30000:]
            #annotations = {}
            #img_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path,'rgb')))]
            # only 32560
            
            mask_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, dataset_name, 'mask')))]
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in mask_idxs]
            if self.set_name == 'trainval_train':
                prefixes = prefixes[:30000]
            elif self.set_name == 'trainval_val':
                prefixes = prefixes[30000:]
            del mask_idxs
            '''
            
            # FOR 32560*4
            img_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, dataset_name, 'rgb')))]
            if self.set_name == 'trainval_train':
                img_idxs = img_idxs[:30000] + img_idxs[32560:30000+32560] + img_idxs[32560*2:30000+32560*2]+ img_idxs[32560*3:30000+32560*3]
            elif self.set_name == 'trainval_val':
                img_idxs = img_idxs[30000:32560] + img_idxs[30000+32560:32560*2] + img_idxs[30000+32560*2:32560*3] + img_idxs[30000+32560*3:]
            
            #elif self.set_name == 'training':
            self.K_list = self.K_list + self.K_list + self.K_list + self.K_list
            self.scale_list = self.scale_list + self.scale_list + self.scale_list + self.scale_list
            self.mano_list = self.mano_list + self.mano_list + self.mano_list + self.mano_list
            self.joint_list = self.joint_list + self.joint_list + self.joint_list + self.joint_list
            self.verts_list = self.verts_list + self.verts_list + self.verts_list + self.verts_list
            self.open_2dj_list = self.open_2dj_list + self.open_2dj_list + self.open_2dj_list + self.open_2dj_list
            self.open_2dj_con_list = self.open_2dj_con_list + self.open_2dj_con_list + self.open_2dj_con_list + self.open_2dj_con_list
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in img_idxs]
            #mask_names = []
            '''
            
        elif self.set_name == 'evaluation':
            img_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, self.set_name, 'rgb')))]
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in img_idxs]
            self.open_2dj_lists = json_load(os.path.join(openpose_v2_path, 'openpose_v2/evaluation', 'detect.json'))
            #self.open_2dj_lists = json_load('/data/FreiHand_save/evaluation/detect.json')
            # self.open_2dj_lists = json_load(os.path.join(self.base_path, 'openpose/evaluation', 'detect.json'))
            self.open_2dj_list = self.open_2dj_lists[0]
            self.open_2dj_con_list = self.open_2dj_lists[1]
            #self.CRFmask_dir = '/data/FreiHand_save/CRFmask/evaluation'
            self.CRFmask_dir = os.path.join(self.base_path, 'CRFmask/evaluation')
        
        image_names = []
        for idx, prefix in enumerate(prefixes):
            image_path = os.path.join(self.base_path, dataset_name, 'rgb', '{}.jpg'.format(prefix))
            #mask_path = os.path.join(self.base_path, 'mask', '{}.jpg'.format(prefix))
            image_names.append(image_path)
            #mask_names.append(mask_path)
        self.image_names = image_names
        del image_names
        del prefixes

    def get_img(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path).convert('RGB')
        #img = func_transforms.to_tensor(img).float()
        return img

    def get_cv_image(self, idx):
        image_path = self.image_names[idx]
        cv_image = cv2.imread(image_path)
        return cv_image
    
    def get_mask(self, idx):
        image_path = self.image_names[idx]
        #mask_path = self.mask_names[idx]
        mask_path = image_path.replace('rgb','mask')
        #mask = cv2.imread(mask_path, 1)
        mask = Image.open(mask_path)
        #mask = func_transforms.to_tensor(mask)
        #mask = torch.round(mask)
        return mask
    
    def get_CRFmask(self, idx):
        CRFmask_path = os.path.join(self.CRFmask_dir,"{:08d}.png".format(idx))
        mask = Image.open(CRFmask_path)
        #mask = func_transforms.to_tensor(mask)
        #mask = torch.round(mask)
        return mask
    
    def get_bgimg(self):
        not_ok_bg = True
        while not_ok_bg:
            bgimg_path = os.path.join(self.bgimgs_dir, choice(self.bgimgs_filename))
            bgimg = Image.open(bgimg_path)
            if len(bgimg.split())==3:
                not_ok_bg=False
        bgimg = func_transforms.resize(bgimg,(224,224))
        return bgimg

    def get_refhand(self):
        not_ok_ref = True
        while not_ok_ref:
            refhand = Image.open(choice(self.refhand_filename))
            if len(refhand.split())==3:
                not_ok_ref=False
        width, height = refhand.size
        if width == 640 and height == 480:#MVHP
            refhand = func_transforms.center_crop(refhand, [400,400])
            
        refhand = func_transforms.rotate(refhand, angle=np.random.randint(-180,180))
        
        refhand = func_transforms.resize(refhand,(224,224))
        return refhand

    def get_maskRGB(self, idx):
        
        image_path = self.image_names[idx]
        img = io.imread(image_path)
        if idx >= 32560:
            idx = idx % 32560
            image_path = self.image_names[idx]
        mask_img =io.imread(image_path.replace('rgb', 'mask'), 1)
        mask_img = np.rint(mask_img)
        #img[~mask_img.astype(bool).repeat(2, 3)] = 0
        img[~mask_img.astype(bool)] = 0
        img = func_transforms.to_tensor(img).float()
        '''
        #img = self.get_img(idx)
        image_path = self.image_names[idx]
        img = io.imread(image_path)
        mask = self.get_mask(idx)
        img[~mask.numpy().astype(bool)]=0
        img = func_transforms.to_tensor(img).float()
        '''
        return img

    def get_K(self, idx):
        K = np.array(self.K_list[idx])
        #K = torch.FloatTensor(K)
        return K
    def get_scale(self, idx):
        scale = self.scale_list[idx]
        return scale
    def get_mano(self,idx):
        mano = self.mano_list[idx]
        mano = torch.FloatTensor(mano)
        return mano
    def get_joint(self,idx):
        joint = self.joint_list[idx]
        joint = torch.FloatTensor(joint)
        return joint
    def get_vert(self, idx):
        verts = self.verts_list[idx]
        verts = torch.FloatTensor(verts)
        return verts
    def get_open_2dj(self, idx):
        open_2dj = self.open_2dj_list[idx]
        open_2dj = torch.FloatTensor(open_2dj)
        return open_2dj
    def get_open_2dj_con(self, idx):
        open_2dj_con = self.open_2dj_con_list[idx]
        open_2dj_con = torch.FloatTensor(open_2dj_con)
        return open_2dj_con
    def __len__(self):
        return len(self.image_names)

def _get_segm(img, side="left"):
    if side == "right":
        hand_segm_img = (img == 22).astype(float) + (img == 24).astype(float)
    elif side == "left":
        hand_segm_img = (img == 21).astype(float) + (img == 23).astype(float)
    else:
        raise ValueError("Got side {}, expected [right|left]".format(side))

    obj_segm_img = (img == 100).astype(float)
    segm_img = np.stack(
        [hand_segm_img, obj_segm_img, np.zeros_like(hand_segm_img)], axis=2
    )
    return segm_img

# DARTset
RAW_IMAGE_SIZE = 512
BG_IMAGE_SIZE = 224 # origin: 384

class DARTset():

    def __init__(self, data_root="/mnt/data/zhuoran", data_split="train", use_full_wrist=False, load_wo_background=False):

        self.name = "DARTset"
        self.data_split = data_split
        self.root = os.path.join(data_root, self.name, self.data_split)
        self.load_wo_background = load_wo_background
        self.raw_img_size = RAW_IMAGE_SIZE
        self.img_size = RAW_IMAGE_SIZE if load_wo_background else BG_IMAGE_SIZE
        self.split = data_split

        self.use_full_wrist = use_full_wrist

        self.MANO_pose_mean = ManoLayer(joint_rot_mode="axisang",
                                        use_pca=False,
                                        mano_assets_root="/home/zhuoran/HandRecon/data/assets/mano_v1_2",
                                        center_idx=0,
                                        flat_hand_mean=False).th_hands_mean.numpy().reshape(-1)

        obj_filename = os.path.join('/home/zhuoran/HandRecon/data/assets/hand_mesh.obj')
        _, faces, _ = load_obj(
            obj_filename,
            device="cpu",
            load_textures=False,
        )
        self.reorder_idx = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
        self.hand_faces = faces[0].numpy()

        self.load_dataset()

    def load_dataset(self):

        self.image_paths = []
        self.raw_mano_param = []
        self.joints_3d = []
        self.verts_3d_paths = []
        self.joints_2d = []

        image_parts = [
            r for r in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, r)) and "verts" not in r and "wbg" not in r
        ]
        image_parts = sorted(image_parts)

        for imgs_dir in image_parts:
            imgs_path = os.path.join(self.root, imgs_dir)
            data_record = pickle.load(open(os.path.join(self.root, f"part_{imgs_dir}.pkl"), "rb"))
            for k in range(len(data_record["pose"])):
                self.image_paths.append(os.path.join(imgs_path, data_record["img"][k]))
                self.raw_mano_param.append(data_record["pose"][k].astype(np.float32))
                self.joints_3d.append(data_record["joint3d"][k].astype(np.float32))
                self.joints_2d.append(data_record["joint2d"][k].astype(np.float32))
                verts_3d_path = os.path.join(imgs_path + "_verts", data_record["img"][k].replace(".png", ".pkl"))
                self.verts_3d_paths.append(verts_3d_path)

        self.sample_idxs = list(range(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            "image": self.get_image(idx),
            "joints_3d": self.get_joints_3d(idx),
            "joints_2d": self.get_joints_2d(idx),
            "joints_uvd": self.get_joints_uvd(idx),
            "verts_uvd": self.get_verts_uvd(idx),
            "ortho_intr": self.get_ortho_intr(idx),
            "sides": self.get_sides(idx),
            "mano_pose": self.get_mano_pose(idx),
            "image_mask": self.get_image_mask(idx),
            "verts_3d": self.get_verts_3d(idx),
            "idxs": idx,
        }

    def get_joints_3d(self, idx):
        joints = self.joints_3d[idx].copy()
        # * Transfer from UNITY coordinate system
        joints[:, 1:] = -joints[:, 1:]
        joints = joints[self.reorder_idx]
        
        # joints = joints - joints[9] + np.array(
        #     [0, 0, 0.5])  # * We use ortho projection, so we need to shift the center of the hand to the origin
        
        # Not to perform joint normalization in dataset stage
        # joints = joints + np.array(
        #     [0, 0, 0.5])  # * We use ortho projection, so we need to shift the center of the hand to the origin
        return joints

    def get_verts_3d(self, idx):
        verts = pickle.load(open(self.verts_3d_paths[idx], "rb"))
        # * Transfer from UNITY coordinate system
        verts[:, 1:] = -verts[:, 1:]
        verts = verts + self.get_joints_3d(idx)[5]
        if not self.use_full_wrist:
            verts = verts[:778]
        verts = verts.astype(np.float32)
        return verts

    def get_joints_2d(self, idx):
        joints_2d = self.joints_2d[idx].copy()[self.reorder_idx]
        joints_2d = joints_2d / self.raw_img_size * self.img_size
        return joints_2d

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_ortho_intr(self, idx):
        ortho_cam = fit_ortho_param(self.get_joints_3d(idx), self.get_joints_2d(idx))
        return ortho_cam

    def get_image(self, idx):
        path = self.image_paths[idx]
        if self.load_wo_background:
            img = np.array(imageio.imread(path, pilmode="RGBA"), dtype=np.uint8)
            img = img[:, :, :3]
        else:
            path = os.path.join(*path.split("/")[:-2], path.split("/")[-2] + "_wbg", path.split("/")[-1])
            path = '/' + path # add '/' to the beginning of  image path
            img = cv2.imread(path)[..., ::-1]
        img = cv2.resize(img, dsize=(self.img_size, self.img_size)) # resize img to 224
        return img

    def get_image_mask(self, idx):
        path = self.image_paths[idx]
        image = np.array(imageio.imread(path, pilmode="RGBA"), dtype=np.uint8)
        image = cv2.resize(image, dsize=(self.img_size, self.img_size))
        return (image[:, :, 3] >= 128).astype(np.float32) * 255.0

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        ortho_cam = self.get_ortho_intr(idx)
        ortho_proj_verts = ortho_project(v3d, ortho_cam)
        d = v3d[:, 2:]
        uvd = np.concatenate((ortho_proj_verts, d), axis=1)
        return uvd

    def get_raw_mano_param(self, idx):
        return self.raw_mano_param[idx].copy()

    def get_mano_pose(self, idx):
        pose = self.get_raw_mano_param(idx)  # [16, 3]

        # * Transfer from UNITY coordinate system
        unity2cam = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32)
        root = rotmat_to_aa(unity2cam @ aa_to_rotmat(pose[0]))[None]
        new_pose = np.concatenate([root.reshape(-1), pose[1:].reshape(-1) + self.MANO_pose_mean], axis=0)  # [48]
        return new_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        return np.zeros((10), dtype=np.float32)

    def get_sides(self, idx):
        return "right"

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

class RHD:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.load_dataset()
        self.name = "RHD"
        self.split = split
    
    # Annotations
    def load_dataset(self):
        set = self.set_name
        # load annotations of this set
        with open(os.path.join(self.base_path, set, 'anno_%s.pickle' % set), 'rb') as fi:
            self.anno_all = pickle.load(fi)
        
        # self.open_2dj_lists = json_load(os.path.join(self.base_path, set, 'openpose', 'detect.json'))
        # self.open_2dj_list = self.open_2dj_lists[0]
        # self.open_2dj_con_list = self.open_2dj_lists[1]
        img_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, self.set_name, 'color')))]
        self.prefix_template = "{:05d}"
        prefixes = [self.prefix_template.format(idx) for idx in img_idxs]
        image_names = []
        for idx, prefix in enumerate(prefixes):
            image_path = os.path.join(self.base_path, self.set_name, 'color', '{}.png'.format(prefix))
            image_names.append(image_path)
        self.image_names = image_names
        del image_names
        del img_idxs
        del prefixes
    
    def get_center_scale(self, idx, scale_factor=2.5):#scale_factor=2.2
        joints2d = self.get_joints2d(idx).astype(np.float32)
        center = handutils.get_annot_center(joints2d)
        scale = handutils.get_annot_scale(
            joints2d, scale_factor=scale_factor
        )
        return center, scale

    # RGB
    def get_img(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path).convert('RGB')
        #img = func_transforms.to_tensor(img).float()
        return img
    
    # mask
    def get_mask(self, idx):
        image_path = self.image_names[idx]
        mask_path = image_path.replace('color','mask')
        
        mask = Image.open(mask_path)
        #mask = func_transforms.to_tensor(mask)
        #mask = mask * 255
        #mask = mask>0
        '''
        import imageio
        mask = imageio.imread(mask_path)
        mask = func_transforms.to_tensor(mask)
        #mask = mask>0
        '''
        return mask#.int()
    
    # depth
    def get_depth(self,idx):
        image_path = self.image_names[idx]
        depth_path = image_path.replace('color','depth')
        #img = cv2.imread(image_path, -1)
        img = Image.open(depth_path)
        '''
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))
        '''
        #img = Image.fromarray(img)#add by cyj
        #img = depth_two_uint8_to_float(img[:, :, 0], img[:, :, 1])
        #img = func_transforms.to_tensor(img).float()
        return img

    # camera intrinsic matrix
    def get_K(self, idx):
        K = np.array(self.anno_all[idx]['K'])
        #K = torch.FloatTensor(K)
        return K
    
    # 3d keypoints coordinates, in meters
    def get_joint(self,idx):
        joint = np.array(self.anno_all[idx]['xyz'])
        #joint = torch.FloatTensor(joint)
        return joint
    
    # 2d keypoints uv
    def get_joints2d(self,idx):
        joints2d = np.array(self.anno_all[idx]['uv_vis'][:, :2])
        #joints2d = torch.FloatTensor(joints2d)
        return joints2d
    
    # visibility of the keypoints, boolean
    def get_j2dvis(self, idx):
        j2d_vis = np.array(self.anno_all[idx]['uv_vis'][:, 2])
        j2d_vis = torch.BoolTensor(j2d_vis)
        return j2d_vis

    def get_open_2dj(self, idx):
        open_2dj = self.open_2dj_list[idx]
        open_2dj = torch.FloatTensor(open_2dj)
        return open_2dj
    
    def get_open_2dj_con(self, idx):
        open_2dj_con = self.open_2dj_con_list[idx]
        open_2dj_con = torch.FloatTensor(open_2dj_con)
        return open_2dj_con
    
    def __len__(self):
        return len(self.image_names)

class HO3D:
    def __init__(
        self,
        set_name=None,
        base_path=None,
        split = 'train',
    ):
        self.set_name = set_name
        self.base_path = base_path
        self.load_dataset()
        self.name = "HO3D"
        self.split = split
        self.cam_extr = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    
    # Annotations
    def load_dataset(self):
        
        #self.K_list = json_load(os.path.join(self.base_path, '%s_K.json' % self.set_name))
        #self.scale_list = json_load(os.path.join(self.base_path, '%s_scale.json' % self.set_name))
        #import pdb; pdb.set_trace()
        
        if self.set_name == 'training' or self.set_name == 'trainval_train' or self.set_name == 'trainval_val':
            training_file = open(os.path.join(self.base_path,'train.txt'),"r") 
            training_list = training_file.readlines()#66034
            if self.set_name == 'trainval_train':
                training_list = training_list[:63000]
            elif self.set_name == 'trainval_val':
                training_list = training_list[63000:]
            
            self.image_list = [i.strip().split('/') for i in training_list]
            self.subfolder = "train"
            # read openpose
            seq_list = os.listdir(os.path.join(self.base_path,'train'))
            self.open_2dj_list={}
            for seq_item in seq_list:
                open_2dj_list = json_load(os.path.join(self.base_path,'openpose',seq_item,'detect.json'))
                self.open_2dj_list[seq_item] = open_2dj_list
            #import pdb; pdb.set_trace()
            for_one_sub = False
            if for_one_sub:
                sub_id = 0
                new_image_list = []
                for image_l in self.image_list:
                    if image_l[0] == seq_list[sub_id]:
                        new_image_list.append(image_l)
                #import pdb; pdb.set_trace()
                print(seq_list[sub_id])
                self.image_list = new_image_list

        elif self.set_name == 'evaluation':
            evaluation_file = open(os.path.join(self.base_path,'evaluation.txt'),"r") 
            evaluation_list = evaluation_file.readlines()#11524
            self.image_list = [i.strip().split('/') for i in evaluation_list]
            self.subfolder = "evaluation"
            # read openpose
            # '''
            seq_list = os.listdir(os.path.join(self.base_path,'evaluation'))
            self.open_2dj_list={}
            for seq_item in seq_list:
                open_2dj_list = json_load(os.path.join(self.base_path,'openpose',seq_item,'detect.json'))
                self.open_2dj_list[seq_item] = open_2dj_list
            # '''
            
        #self.cam_intr = np.array([[617.343, 0.0, 312.42], [0.0, 617.343, 241.42], [0.0, 0.0, 1.0]]).astype(np.float32)
        #self.cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        #import pdb;pdb.set_trace()

    def get_img(self, idx):
        [seq_name, image_name] = self.image_list[idx]
        image_path = os.path.join(self.base_path, self.subfolder, seq_name, 'rgb', '{}.png'.format(image_name))
        img = Image.open(image_path).convert('RGB')
        #img = func_transforms.resize(img,64)
        #img = func_transforms.to_tensor(img).float()#[3,480,640]
        return img

    
    def get_masks(self, idx):
        [seq_name, image_name] = self.image_list[idx]
        mask_path = os.path.join(self.base_path, self.subfolder, seq_name, 'seg', '{}.jpg'.format(image_name))
        mask = Image.open(mask_path)
        mask = func_transforms.resize(mask,(480,640))
        #img = func_transforms.resize(img,64)
        #mask = func_transforms.to_tensor(mask)#[3,120,160] 0~1
        #mask = torch.round(mask)
        return mask
    

    def get_depth(self, idx):
        [seq_name, image_name] = self.image_list[idx]
        depth_path = os.path.join(self.base_path, self.subfolder, seq_name, 'depth', '{}.png'.format(image_name))
        depth = Image.open(depth_path)
        #depth = func_transforms.to_tensor(depth)#[3,480,640] 0~1
        '''
        depth_scale = 0.00012498664727900177
        depth_img = cv2.imread(depth_path)
        dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
        dpt = dpt * depth_scale
        #[480,640]
        return dpt
        '''
        return depth
    
    def get_meta(self, idx):
        [seq_name, image_name] = self.image_list[idx]
        meta_path = os.path.join(self.base_path, self.subfolder, seq_name, 'meta', '{}.pkl'.format(image_name))
        #meta = pickle.load(meta_path)
        meta = pickle_load(meta_path)
        # meta.keys() dict_keys(['objTrans', 'handBeta', 'camMat', 'handJoints3D', 'handTrans', 'camIDList', 'handPose', 'objCorners3DRest', 'objRot', 'objName', 'objCorners3D', 'objLabel'])
        
        # evaluation
        # dict_keys(['camIDList', 'objTrans', 'objCorners3DRest', 'objLabel', 'handBoundingBox', 'objCorners3D', 'objName', 'camMat', 'handJoints3D', 'objRot'])
        
        return meta
    
    def get_open_2dj(self, idx):
        #self.open_2dj = json_load('/data/HO3D/train/SS1/openpose/detect.json')
        [seq_name, image_name] = self.image_list[idx]
        open_2dj = self.open_2dj_list[seq_name][0][int(image_name)]
        open_2dj = torch.FloatTensor(open_2dj)
        return open_2dj

    def get_open_2dj_con(self, idx):
        [seq_name, image_name] = self.image_list[idx]
        open_2dj_con = self.open_2dj_list[seq_name][1][int(image_name)]
        open_2dj_con = torch.FloatTensor(open_2dj_con)
        return open_2dj_con

    def __len__(self):
        return len(self.image_list)    
