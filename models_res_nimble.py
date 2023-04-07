import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import time

import numpy as np
import pytorch3d
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
import pytorch3d.renderer as p3d_renderer
from network.res_encoder import ResEncoder, HandEncoder, LightEstimator
from utils.NIMBLE_model.myNIMBLELayer import MyNIMBLELayer
from utils.traineval_util import Mano2Frei, trans_proj_j2d
from utils.my_mano import MyMANOLayer
from utils.Freihand_GNN_mano.mano_network_PCA import YTBHand
from utils.Freihand_GNN_mano.Freihand_trainer_mano_fullsup import dense_pose_Trainer
ytbHand_trainer = dense_pose_Trainer(None, None)


class Model(nn.Module):
    def __init__(self, ifRender, device, if_4c, hand_model, use_mean_shape, pretrain, root_id=9, root_id_nimble=11):
        super(Model, self).__init__()
        self.hand_model = hand_model
        self.root_id = root_id
        self.root_id_nimble = root_id_nimble
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
            # Define a renderer in pytorch3d
            # Initialize a perspective camera.
            cameras = p3d_renderer.cameras.PerspectiveCameras(device=device)
            # Rasterization settings for differentiable rendering, where the blur_radius
            # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019
            sigma = 1e-4
            raster_settings_soft = RasterizationSettings(
                image_size=224, 
                # blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
                blur_radius=0.0, 
                # blur_radius=0.002, 
                faces_per_pixel=1, 
                # perspective_correct=False, 
            )

            # # Differentiable soft renderer using per vertex RGB colors for texture
            # renderer_textured = MeshRenderer(
            #     rasterizer=MeshRasterizer(
            #         cameras=camera, 
            #         raster_settings=raster_settings_soft
            #     ),
            #     shader=SoftPhongShader(device=device, 
            #         cameras=camera,
            #         lights=lights)
            # create a renderer object
            self.renderer_p3d = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings_soft),
                shader=SoftPhongShader(device=device, cameras=cameras),
            )




    def forward(self, images, Ks=None, scale_gt=None, root_xyz=None):
        if self.hand_model == 'mano_new':
            pred = self.ytbHand(images)
            outputs = {
                'pose_params': pred['theta'],
                'shape_params': pred['beta'],
                'verts': pred['mesh']
                }
            return outputs
        device = images.device
        batch_size = images.shape[0]
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
        outputs.update(hand_params)           

        # map nimble 25 joints to freihand 21 joints
        if self.hand_model == 'mano_new':
            # regress joints from verts
            vertice_pred_list = outputs['verts']
            outputs['joints'] = ytbHand_trainer.xyz_from_vertice(vertice_pred_list[-1]).permute(1,0,2)
        elif self.hand_model == 'mano':
            # regress joints from verts
            vertice_pred_list = outputs['mano_verts']
            outputs['joints'] = ytbHand_trainer.xyz_from_vertice(vertice_pred_list).permute(1,0,2)
        else: # nimble
            # Mano joints map to Frei joints
            outputs['joints'] = Mano2Frei(outputs['joints'])

        # ** offset positions relative to root.
        pred_root_xyz = outputs['joints'][:, self.root_id, :].unsqueeze(1)
        outputs['joints'] = outputs['joints'] - pred_root_xyz
        outputs['mano_verts'] = outputs['mano_verts'] - pred_root_xyz
        if self.hand_model == 'nimble':
            pred_root_xyz = outputs['nimble_joints'][:, self.root_id_nimble, :].unsqueeze(1)
            outputs['nimble_joints'] = outputs['nimble_joints'] - pred_root_xyz


        # Render image
        if self.ifRender:
            # set up renderer parameters
            
            k_44 = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            k_44[:, :3, :4] = Ks
            cameras = p3d_renderer.cameras.PerspectiveCameras(K=k_44, device=device, in_ndc=False, image_size=((224,224),)) # R and t are identity and zeros by default
            # cameras = p3d_renderer.cameras.PerspectiveCameras(K=k_44, device=device) # R and t are identity and zeros by default
            lighting = p3d_renderer.lighting.PointLights(
                # ambient_color=((1.0, 1.0, 1.0),),
                # diffuse_color=((0.0, 0.0, 0.0),),
                # specular_color=((0.0, 0.0, 0.0),),
                # location=((0.0, 0.0, 0.0),),
                device=device,
            )

            # render the image
            # move to the root relative coord. verts = verts - pred_root_xyz + root_xyz
            verts_num = outputs['skin_meshes']._num_verts_per_mesh[0]
            outputs['skin_meshes'].offset_verts_(-pred_root_xyz.repeat(1, verts_num, 1).view(verts_num*batch_size, 3))
            outputs['skin_meshes'].offset_verts_(root_xyz.repeat(1, verts_num, 1).view(verts_num*batch_size, 3))
            rendered_images = self.renderer_p3d(outputs['skin_meshes'], cameras=cameras, lights=lighting)

            # import torchvision
            # torchvision.utils.save_image(rendered_images[...,:3][1].permute(2,0,1),"test.png")

            outputs['re_img'] = rendered_images[..., :3]

        return outputs
    