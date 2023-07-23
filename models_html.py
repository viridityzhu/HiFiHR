import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import math
import time

import numpy as np
import pytorch3d
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,
    HardPhongShader,
    Materials
)
from pytorch3d.renderer.lighting import DirectionalLights
import pytorch3d.renderer as p3d_renderer
from network.res_encoder import ResEncoder, TexEncoder, LightEstimator
from network.effnet_encoder import EffiEncoder
from utils.NIMBLE_model.myNIMBLELayer import MyNIMBLELayer
from utils.traineval_util import Mano2Frei, trans_proj_j2d
from utils.my_html import MyHTMLLayer
from utils.my_mano import MyMANOLayer
from utils.Freihand_GNN_mano.Freihand_trainer_mano_fullsup import dense_pose_Trainer
ytbHand_trainer = dense_pose_Trainer(None, None)


class Model(nn.Module):
    def __init__(self, device, if_4c, pretrain, root_id=9, root_id_nimble=11):
        super(Model, self).__init__()
        self.root_id = root_id
        self.root_id_nimble = root_id_nimble

        if pretrain == 'hr18sv2':
            self.features_dim = 1024 # for HRnet
            self.low_feat_dim = 512 # not sure
            self.base_encoder = ResEncoder(pretrain=pretrain, if_4c=if_4c)
        elif pretrain in ['res18', 'res50', 'res101']:
            self.features_dim = 2048
            self.low_feat_dim = 512
            self.base_encoder = ResEncoder(pretrain=pretrain, if_4c=if_4c)
        elif pretrain == 'effb3':
            self.features_dim = 1536
            self.low_feat_dim = 32
            self.base_encoder = EffiEncoder(pretrain=pretrain)
        
        self.ncomps = 101 # texture.
        self.hand_layer = MyHTMLLayer(device, tex_ncomp=self.ncomps)
            
        self.tex_encoder = TexEncoder(ncomps=self.ncomps, in_dim=self.features_dim)

        self.aa_factor = 3
        # Renderer
        # Define a RGB renderer with HardPhongShader in pytorch3d
        raster_settings_soft = RasterizationSettings(
            image_size=224 * self.aa_factor, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        materials = Materials(
            # ambient_color=((0.9, 0.9, 0.9),),
            diffuse_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            shininess=30,
            device=device,
        )

        # Differentiable soft renderer with SoftPhongShader
        self.renderer_p3d = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_soft
            ),
            shader=HardPhongShader(
                materials=materials,
                device=device, 
            ),
        )

        self.light_estimator = LightEstimator(self.low_feat_dim)


    def forward(self, dat_name, mode_train, model_prev, outputs_prev, images, Ks=None, root_xyz=None):
        device = images.device
        batch_size = images.shape[0]
        # Use base_encoder to extract features
        # low_features, features = self.base_encoder(images) # [b, 512, 14, 14], [b,1024]
        
        low_features, features = model_prev.module.base_encoder(images) # [b, 512, 14, 14], [b,1024]
        
        # light_params = self.light_estimator(low_features)

        # Use hand_encoder to get hand parameters
        tex_params  = self.tex_encoder(features)
        # tex_params = {
        #     'texture_params': texture_params, 
        # }

        # Use nimble_layer to get 3D hand models
        outputs = self.hand_layer(tex_params, outputs_prev['mano_verts'], outputs_prev['mano_faces'])
        ######### outputs = {
        #########     'nimble_joints': bone_joints, # 25 joints
        #########     'verts': skin_v, # 5990 verts
        #########     'faces': None #faces,
        #########     'skin_meshes': skin_v_smooth, # smoothed verts and faces
        #########     'mano_verts': skin_mano_v, # 5990 -> 778 verts according to mano
        #########     'textures': tex_img,
        #########     'rot':rot
        ######### }
        # outputs = {
        #     'skin_meshes': skin_v_smooth, # smoothed verts and faces
        #     'textures': tex_img,
        # }
        outputs.update(tex_params)           


        # Render image
        # set up renderer parameters
        # k_44 = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        # k_44[:, :3, :4] = Ks
        # cameras = p3d_renderer.cameras.PerspectiveCameras(K=k_44, device=device, in_ndc=False, image_size=((224,224),)) # R and t are identity and zeros by default

        # get ndc fx, fy, cx, cy from Ks
        fcl, prp = self.get_ndc_fx_fy_cx_cy(Ks)
        cameras = p3d_renderer.cameras.PerspectiveCameras(focal_length=-fcl, 
                                                            principal_point=prp,
                                                            device=device) # R and t are identity and zeros by default
        lighting = p3d_renderer.lighting.PointLights(
            # ambient_color=((1.0, 1.0, 1.0),),
            # diffuse_color=((0.0, 0.0, 0.0),),
            # specular_color=((0.0, 0.0, 0.0),),
            # location=((0.0, 0.0, 0.0),),
            device=device,
        )


        # move to the root relative coord. 
        # verts = verts - pred_root_xyz + root_xyz
        # verts_num = outputs['skin_meshes']._num_verts_per_mesh[0]
        # outputs['skin_meshes'].offset_verts_(-pred_root_xyz.repeat(1, verts_num, 1).view(verts_num*batch_size, 3))
        # outputs['skin_meshes'].offset_verts_(root_xyz.repeat(1, verts_num, 1).view(verts_num*batch_size, 3))

        # render the image
        rendered_images = self.renderer_p3d(outputs['skin_meshes'], cameras=cameras, lights=lighting)
        # average pooling to downsample the rendered image (anti-aliasing)
        rendered_images = rendered_images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        rendered_images = F.avg_pool2d(rendered_images, kernel_size=self.aa_factor, stride=self.aa_factor)
        # rendered_images = rendered_images.permute(0, 2, 3, 1)  # NCHW -> NHWC

        # import torchvision
        # torchvision.utils.save_image(rendered_images[...,:3][1].permute(2,0,1),"test.png")

        outputs['re_img'] = rendered_images[:, :3, :, :] # the last dim is alpha
        outputs['re_sil'] = rendered_images[:, 3:4, :, :] # [B, 1, w, h]. the last dim is alpha
        outputs['re_sil'][outputs['re_sil'] > 0] = 255  # Binarize segmentation mask
        outputs['maskRGBs'] = images.mul((outputs['re_sil']>0).float().repeat(1,3,1,1))
        
        # add mano faces to outputs (used in losses)
        # outputs['mano_faces'] = self.mano_face.repeat(batch_size, 1, 1)

        return outputs
    
    # get ndc fx, fy, cx, cy from Ks
    def get_ndc_fx_fy_cx_cy(self, Ks):
        ndc_fx = Ks[:, 0, 0] * 2 / 224.0
        ndc_fy = Ks[:, 1, 1] * 2 / 224.0
        ndc_px = - (Ks[:, 0, 2] - 112.0) * 2 / 224.0
        ndc_py = - (Ks[:, 1, 2] - 112.0) * 2 / 224.0
        focal_length = torch.stack([ndc_fx, ndc_fy], dim=-1)
        principal_point = torch.stack([ndc_px, ndc_py], dim=-1)
        return focal_length, principal_point
        