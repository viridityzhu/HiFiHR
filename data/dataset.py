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
            #self.open_2dj_lists = json_load('/data/FreiHand_save/evaluation/detect.json')
            # self.open_2dj_lists = json_load(os.path.join(self.base_path, 'openpose/evaluation', 'detect.json'))
            self.open_2dj_lists = json_load(os.path.join(openpose_v2_path, 'openpose_v2/evaluation', 'detect.json'))
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
        joints = joints + np.array(
            [0, 0, 0.5])  # * We use ortho projection, so we need to shift the center of the hand to the origin
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