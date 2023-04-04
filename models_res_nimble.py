import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import time

import numpy as np
import pytorch3d
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
from network.res_encoder import ResEncoder, HandEncoder, LightEstimator
from utils.NIMBLE_model.myNIMBLELayer import MyNIMBLELayer
from utils.my_mano import MyMANOLayer
from utils.Freihand_GNN_mano.mano_network_PCA import YTBHand


class Model(nn.Module):
    def __init__(self, ifRender, device, if_4c, hand_model, use_mean_shape, pretrain):
        super(Model, self).__init__()
        self.hand_model = hand_model
        if hand_model == 'mano_new':
            self.ytbHand = YTBHand(None, None, use_pca=True, pca_comps=48)
            return

        if pretrain == 'hr18sv2':
            self.features_dim = 1024 # for HRnet
        elif pretrain in ['res18', 'res50', 'res101']:
            self.features_dim = 2048
        self.base_encoder = ResEncoder(pretrain=pretrain, if_4c=if_4c)

        if hand_model == 'nimble':
            self.ncomps = [20, 30, 10] # shape, pose, tex respectively.
            self.hand_layer = MyNIMBLELayer(ifRender, device, shape_ncomp=self.ncomps[0], pose_ncomp=self.ncomps[1], tex_ncomp=self.ncomps[2])
        elif hand_model == 'mano':
            self.ncomps = [10, 48, None] # shape, pose, no texture.
            self.hand_layer = MyMANOLayer(ifRender, device, shape_ncomp=self.ncomps[0], pose_ncomp=self.ncomps[1], tex_ncomp=self.ncomps[2])
            
        self.hand_encoder = HandEncoder(hand_model=hand_model, ncomps=self.ncomps, in_dim=self.features_dim, ifRender=ifRender, use_mean_shape=use_mean_shape)


        self.ifRender = ifRender

        # Renderer
        if self.ifRender:
            pass
            # Define a renderer in pytorch3d
            # Rasterization settings for differentiable rendering, where the blur_radius
            # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019
            # sigma = 1e-4
            # raster_settings_soft = RasterizationSettings(
            #     image_size=224, 
            #     blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            #     faces_per_pixel=100, 
            #     # perspective_correct=False, 
            # )

            # # Differentiable soft renderer using per vertex RGB colors for texture
            # renderer_textured = MeshRenderer(
            #     rasterizer=MeshRasterizer(
            #         cameras=camera, 
            #         raster_settings=raster_settings_soft
            #     ),
            #     shader=SoftPhongShader(device=device, 
            #         cameras=camera,
            #         lights=lights)



    def forward(self, images, Ks=None, scale_gt=None):
        if self.hand_model == 'mano_new':
            pred = self.ytbHand(images)
            outputs = {
                'pose_params': pred['theta'],
                'shape_params': pred['beta'],
                'verts': pred['mesh']
                }
            return outputs
        device = images.device
        # Use base_encoder to extract features
        # low_features, features = self.base_encoder(images) # [b, 512, 14, 14], [b,1024]
        _, features = self.base_encoder(images) # [b, 512, 14, 14], [b,1024]
        
        # Use hand_encoder to get hand parameters
        hand_params  = self.hand_encoder(features)
        # hand_params = {
        #     'pose_params': pose_params, 
        #     'shape_params': shape_params, 
        #     'texture_params': texture_params, 
        #     'scale': scale, 
        #     'trans': trans, 
        #     'rot': rot # only for mano hand model
        # }

        # Use nimble_layer to get 3D hand models
        outputs = self.hand_layer(hand_params, handle_collision=False)
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
            pass
            # re_img = self.renderer(outputs['skin_meshes'], cameras=cameras, lights=lights)
        outputs.update(hand_params)           

        return outputs
    
    def render(self, meshes, faces, textures):
        # Given mesh, face and texture, render the image using PyTorch3D
        pass
