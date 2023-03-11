import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import time

import numpy as np
from network.res_encoder import ResEncoder, HandEncoder, LightEstimator
from utils.NIMBLE_model.myNIMBLELayer import MyNIMBLELayer


class Model(nn.Module):
    def __init__(self, ifRender, device, if_4c, hand_model):
        super(Model, self).__init__()

        self.features_dim = 1024 # for HRnet
        self.base_encoder = ResEncoder(pretrain='hr18sv2', if_4c=if_4c)

        if hand_model == 'nimble':
            self.ncomps = [20, 30, 10] # shape, pose, tex respectively.
            self.nimble_layer = MyNIMBLELayer(ifRender, device, shape_ncomp=self.ncomps[0], pose_ncomp=self.ncomps[1], tex_ncomp=self.ncomps[2])
        elif hand_model == 'mano':
            self.ncomps = [10, 30, None] # shape, pose, no texture.
            self.nimble_layer = MyMANOLayer(ifRender, device, shape_ncomp=self.ncomps[0], pose_ncomp=self.ncomps[1], tex_ncomp=self.ncomps[2])
            
        self.hand_encoder = HandEncoder(hand_model=hand_model, ncomps=self.ncomps, in_dim=self.features_dim, ifRender=ifRender)


        self.ifRender = ifRender

        # Renderer & Texture Estimation & Light Estimation
        if self.ifRender:
            pass # TODO: define the renderer
            # Define a neural renderer
            # import neural_renderer as nr
            # if 'lights' in args.train_requires or 'lights' in args.test_requires:
            #     renderer_NR = nr.Renderer(image_size=args.image_size,background_color=[1,1,1],camera_mode='projection',orig_size=224,light_intensity_ambient=None, light_intensity_directional=None,light_color_ambient=None, light_color_directional=None,light_direction=None)#light_intensity_ambient=0.9
            # else:
            #     renderer_NR = nr.Renderer(image_size=args.image_size,camera_mode='projection',orig_size=224)
            # self.renderer_NR = renderer_NR

            '''
            if self.texture_choice == 'surf':
                self.texture_estimator = TextureEstimator(dim_in=self.dim_in,mode='surfaces')
            elif self.texture_choice == 'nn_same':
                self.color_estimator = ColorEstimator(dim_in=self.dim_in)
            self.light_estimator = light_estimator(dim_in=self.dim_in)
            '''
            # self.light_estimator = LightEstimator(mode='surf')

        # Pose adapter
        # if (args.train_datasets)[0] == 'FreiHand':
        #     self.get_gt_depth = True
        #     self.dataset = 'FreiHand'
        # # elif (args.train_datasets)[0] == 'RHD':
        # #     self.get_gt_depth = False
        # #     self.dataset = 'RHD'
        # #     if self.use_pose_regressor: # by default false
        # #         self.mesh2pose = mesh2poseNet()
        # # elif (args.train_datasets)[0] == 'Obman':
        # #     self.get_gt_depth = False
        # #     self.dataset = 'Obman'
        # elif (args.train_datasets)[0] == 'HO3D':
        #     self.get_gt_depth = True
        #     self.dataset = 'HO3D'
        # else:
        #     self.get_gt_depth = False




    def forward(self, images, Ks=None, scale_gt=None):
        device = images.device
        # Use base_encoder to extract features
        low_features, features = self.base_encoder(images) # [b, 512, 14, 14], [b,1024]
        
        # Use hand_encoder to get hand parameters
        hand_params  = self.hand_encoder(features)
        # hand_params = {
        #     'pose_params': pose_params, 
        #     'shape_params': shape_params, 
        #     'texture_params': texture_params, 
        #     'scale': scale, 
        #     'trans': trans, 
        # }

        # Use nimble_layer to get 3D hand models
        outputs = self.nimble_layer(hand_params, handle_collision=False, scale_gt=scale_gt)
        # outputs = {
        #     'nimble_joints': bone_joints, # 25 joints
        #     'verts': skin_v, # 5990 verts
        #     'faces': faces,
        #     'skin_meshes': skin_v_smooth, # smoothed verts and faces
        #     'mano_verts': skin_mano_v, # 5990 -> 778 verts according to mano
        #     'textures': tex_img,
        #     'rot':rot
        # }

        # map nimble 25 joints to freihand 21 joints

        if self.ifRender:
            pass
            # TODO: implement rendering using tex_img
            # lights = self.light_estimator(low_features)
            # self.renderer_NR.light_intensity_ambient = lights[:,0].to(device)
            # self.renderer_NR.light_intensity_directional = lights[:,1].to(device)
            # self.renderer_NR.light_color_ambient = lights[:,2:5].to(device)
            # self.renderer_NR.light_color_directional = lights[:,5:8].to(device)
            # self.renderer_NR.light_direction = lights[:,8:].to(device)
            # outputs['lights'] = lights

            # faces = outputs['faces'].type(torch.int32)
            # # use neural renderer
            # #I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
            # #Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).to(Ks.device)
            
            # self.renderer_NR.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ks.shape[0],1,1).to(device)
            # self.renderer_NR.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ks.shape[0],1,1).to(device)
            # self.renderer_NR.K = Ks[:,:,:3].to(device)
            # self.renderer_NR.dist_coeffs = self.renderer_NR.dist_coeffs.to(device)
            
            # face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
            
            # re_img,re_depth,re_sil = self.renderer_NR(vertices, faces, torch.tanh(face_textures), mode=None)

            # re_depth = re_depth * (re_depth < 1).float()#set 100 into 0

            # if self.get_gt_depth and gt_verts is not None:
            #     gt_depth = self.renderer_NR(gt_verts, faces, mode='depth')
            #     gt_depth = gt_depth * (gt_depth < 1).float()#set 100 into 0
        outputs.update(hand_params)           

        return outputs
