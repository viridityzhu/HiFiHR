'''
    NIMBLE: A Non-rigid Hand Model with Bones and Muscles[SIGGRAPH-22]
    https://reyuwei.github.io/proj/nimble
'''

import os
import numpy as np
import torch
import trimesh
from utils.NIMBLE_model.utils import batch_to_tensor_device, smooth_mesh
from utils.NIMBLE_model.utils import *
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Textures
from torch.autograd import Variable
import pickle

class MyNIMBLELayer(torch.nn.Module):
    __constants__ = [
        'use_pose_pca', 'shape_ncomp', 'pose_ncomp', 'pm_dict'
    ]
    def __init__(self, ifRender, device, use_mean_shape=False, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, pm_dict_name="utils/NIMBLE_model/assets/NIMBLE_DICT_9137.pkl", tex_dict_name="utils/NIMBLE_model/assets/NIMBLE_TEX_DICT.pkl", nimble_mano_vreg_name="utils/NIMBLE_model/assets/NIMBLE_MANO_VREG.pkl", use_pose_pca=True):
        super(MyNIMBLELayer, self).__init__()
        self.device = device
        self.ifRender = ifRender
        self.use_mean_shape = use_mean_shape

        if os.path.exists(pm_dict_name):
            pm_dict = np.load(pm_dict_name, allow_pickle=True)
            pm_dict = batch_to_tensor_device(pm_dict, self.device)

        if os.path.exists(tex_dict_name):
            tex_dict = np.load(tex_dict_name, allow_pickle=True)
            tex_dict = batch_to_tensor_device(tex_dict, self.device)

        if os.path.exists(nimble_mano_vreg_name):
            nimble_mano_vreg = np.load(nimble_mano_vreg_name, allow_pickle=True)
            nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, self.device)
        else:
            nimble_mano_vreg=None
        
        uvs_name = 'utils/NIMBLE_model/assets/faces_uvs.pt'
        if os.path.exists(uvs_name):
            self.register_buffer("faces_uv", torch.load(uvs_name))
            self.register_buffer("verts_uv", torch.load(uvs_name.replace('faces', 'verts')))

        identity_rot = torch.eye(3).to(self.device)
        self.register_buffer("identity_rot", identity_rot)
        
        self.shape_ncomp = shape_ncomp
        self.pose_ncomp = pose_ncomp
        self.tex_ncomp = tex_ncomp
        self.use_pose_pca = use_pose_pca
        self.tex_size = 1024

        self.bone_v_sep = pm_dict['bone_v_sep']
        self.skin_v_sep = pm_dict['skin_v_sep']

        self.register_buffer("th_verts", pm_dict['vert'].squeeze())
        self.register_buffer("th_faces", pm_dict['face'].squeeze())
        self.register_buffer("sw", pm_dict['all_sw'].squeeze())
        self.register_buffer("pbs", pm_dict['all_pbs'].squeeze())
        self.register_buffer("jreg_mano", pm_dict['jreg_mano'].squeeze()) # [21, 5990]
        self.register_buffer("jreg_bone", pm_dict['jreg_bone'].squeeze())
        self.register_buffer("shape_basis", pm_dict['shape_basis'].squeeze())
        self.register_buffer("shape_pm_std", pm_dict['shape_pm_std'].squeeze())
        self.register_buffer("shape_pm_mean", pm_dict['shape_pm_mean'].squeeze())
        self.register_buffer("pose_basis", pm_dict['pose_basis'].squeeze())
        self.register_buffer("pose_mean", pm_dict['pose_mean'].squeeze())
        self.register_buffer("pose_pm_std", pm_dict['pose_pm_std'].squeeze())
        self.register_buffer("pose_pm_mean", pm_dict['pose_pm_mean'].squeeze())

        self.register_buffer("tex_diffuse_basis", tex_dict['diffuse']['basis'].squeeze())
        self.register_buffer("tex_diffuse_mean", tex_dict['diffuse']['mean'].squeeze())
        self.register_buffer("tex_diffuse_std", tex_dict['diffuse']['std'].squeeze())
        self.register_buffer("tex_normal_basis", tex_dict['normal']['basis'].squeeze())
        self.register_buffer("tex_normal_mean", tex_dict['normal']['mean'].squeeze())
        self.register_buffer("tex_normal_std", tex_dict['normal']['std'].squeeze())
        self.register_buffer("tex_spec_basis", tex_dict['spec']['basis'].squeeze())
        self.register_buffer("tex_spec_mean", tex_dict['spec']['mean'].squeeze())
        self.register_buffer("tex_spec_std", tex_dict['spec']['std'].squeeze())

        self.register_buffer("bone_f", pm_dict['bone_f'])
        self.register_buffer("muscle_f", pm_dict['muscle_f'])
        self.register_buffer("skin_f", pm_dict['skin_f'])

        self.skin_v_surface_mask = pm_dict['skin_v_surface_mask'].type(torch.bool)
        self.skin_v_node_weight = dis_to_weight(pm_dict['skin_v_gd'], 30, 50)

        self.th_v_shaped_mean = None
        self.tex_mean = None

        if nimble_mano_vreg is not None:
            self.register_buffer("nimble_mano_vreg_fidx", nimble_mano_vreg['lmk_faces_idx'])
            self.register_buffer("nimble_mano_vreg_bc", nimble_mano_vreg['lmk_bary_coords'])
        else:
            assert "nimble_mano_vreg is None!!" 

        # Kinematic chain params
        kinetree = JOINT_PARENT_ID_DICT
        self.kintree_parents = []
        for i in range(STATIC_JOINT_NUM):
            self.kintree_parents.append(kinetree[i])

    @property
    def bone_v(self):
        bone_v = self.th_verts[:,:self.bone_v_sep,:]
        return bone_v
   
    @property
    def muscle_v(self):
        muscle_v = self.th_verts[:,self.bone_v_sep:self.skin_v_sep,:]
        return muscle_v
  
    @property
    def skin_v(self):
        skin_v = self.th_verts[:,self.skin_v_sep:,:]
        return skin_v
    

    def nimble_to_mano(self, verts, is_surface=False):
        skin_f = self.skin_f # faces
        if not is_surface:
            skin_v = verts[:,self.skin_v_sep:,:]
        else:
            skin_v = verts

        nimble_mano = torch.cat([vertices2landmarks(skin_v, skin_f.squeeze(), self.nimble_mano_vreg_fidx[i],  self.nimble_mano_vreg_bc[i]).unsqueeze(0) for i in range(20)])
        nimble_mano_v = nimble_mano.mean(0)
        return nimble_mano_v

    def compute_warp(self, batch_size, points, skinning_weights, full_trans_mat):
        if points.shape[0] != batch_size:
            points = points.repeat(batch_size, 1, 1)
        if skinning_weights.shape[0] != batch_size:
            skinning_weights = skinning_weights.repeat(batch_size, 1, 1)

        th_T = torch.einsum('bijk,bkt->bijt',full_trans_mat, skinning_weights.permute(0, 2, 1))
        th_rest_shape_h = torch.cat([points.transpose(2, 1),
                                     torch.ones((batch_size, 1, points.shape[1]), dtype=skinning_weights.dtype,
                                                device=skinning_weights.device), ], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        return th_verts

    def generate_hand_shape(self, betas, normalized=True, use_mean_shape=False):
        # beta : B, N
        batch_size, shape_ncomp = betas.shape
        if use_mean_shape and self.th_v_shaped_mean is not None:
            return self.th_v_shaped_mean.unsqueeze(0).repeat(batch_size, 1, 1), self.jreg_bone_joints_mean.unsqueeze(0).repeat(batch_size, 1, 1)
        assert self.shape_ncomp == shape_ncomp

        if normalized:
            betas_real = betas * self.shape_pm_std[:shape_ncomp].reshape(1, -1) + self.shape_pm_mean[:shape_ncomp].reshape(1, -1)
        else:
            betas_real = betas
        th_v_shaped = (self.shape_basis[:shape_ncomp].T @ betas_real.T).view(-1, 3, batch_size).permute(2, 0, 1) + self.th_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        
        jreg_bone_joints = torch.matmul(self.jreg_bone, th_v_shaped[:, :self.bone_v_sep])
        self.th_v_shaped_mean, self.jreg_bone_joints_mean = th_v_shaped[0], jreg_bone_joints[0]
        return th_v_shaped, jreg_bone_joints

    def generate_full_pose(self, theta, normalized=True, with_root=True):
        # theta : B, N
        batch_size = theta.shape[0]

        if with_root:
            real_theta = theta[:, 3:]
            root_rot = theta[:, :3]
        else:
            real_theta = theta
            root_rot = torch.zeros([batch_size, 3]).to(theta.device)

        pose_ncomp = real_theta.shape[-1]
        if normalized:
            theta_real_denorm = real_theta * self.pose_pm_std[:pose_ncomp].reshape(1, -1) + self.pose_pm_mean[:pose_ncomp].reshape(1, -1)
        else:
            theta_real_denorm = real_theta

        full_pose = (self.pose_basis[:pose_ncomp].T @ theta_real_denorm.T).T + self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        full_pose = torch.cat([root_rot, full_pose], dim=1).view(batch_size, -1, 3)

        return full_pose

    def generate_texture(self, alpha, need=True, normalized=True):
        batch_size = alpha.shape[0]
        if not need and self.tex_mean is not None:
            return self.tex_mean.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
        assert self.tex_ncomp == alpha.shape[1]

        if normalized:
            alpha_real_d = alpha * self.tex_diffuse_std[:self.tex_ncomp].reshape(1, -1)
            alpha_real_n = alpha * self.tex_normal_std[:self.tex_ncomp].reshape(1, -1)
            alpha_real_s = alpha * self.tex_spec_std[:self.tex_ncomp].reshape(1, -1)

        x_d = (self.tex_diffuse_basis[:, :self.tex_ncomp] @ alpha_real_d.T).T + self.tex_diffuse_mean.unsqueeze(0).repeat(batch_size, 1)
        x_d = x_d.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x_n = (self.tex_normal_basis[:, :self.tex_ncomp] @ alpha_real_n.T).T + self.tex_normal_mean.unsqueeze(0).repeat(batch_size, 1)
        x_n = x_n.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x_s = (self.tex_spec_basis[:, :self.tex_ncomp] @ alpha_real_s.T).T + self.tex_spec_mean.unsqueeze(0).repeat(batch_size, 1)
        x_s = x_s.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x = torch.cat([x_d, x_n, x_s], dim=-1) # diffuse, normal, and spec
        x = torch.clamp(x, min=0, max=1)

        self.tex_mean = x[0] # [:,:,:, :3]
        return x

    def mano_v2j_reg(self, mano_verts):#[b,778,3]

        batch_size = mano_verts.shape[0]
        MANO_file = 'data/MANO_RIGHT.pkl'
        dd = pickle.load(open(MANO_file, 'rb'),encoding='latin1')
        J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).to(device=mano_verts.device))
        
        # J_reg: [1, 16, 778]
        # [b, 3, 778] x [b, 778, 16] -> [b, 3, 16]
        Jtr = torch.matmul(mano_verts.permute(0,2,1), J_regressor.repeat(batch_size,1,1).permute(0,2,1))
        Jtr = Jtr.permute(0, 2, 1) # b, 16, 3

        # v is of shape: b, 3 (or more) dims, 778 samples
        # For FreiHand: add 5 joints.
        # Jtr.insert(4,mano_verts[:,:3,320].unsqueeze(2))
        # Jtr.insert(8,mano_verts[:,:3,443].unsqueeze(2))
        # Jtr.insert(12,mano_verts[:,:3,672].unsqueeze(2))
        # Jtr.insert(16,mano_verts[:,:3,555].unsqueeze(2))
        # Jtr.insert(20,mano_verts[:,:3,744].unsqueeze(2))      
        Jtr = torch.cat([Jtr[:,:4], mano_verts[:,320].unsqueeze(1), Jtr[:,4:]], 1)
        Jtr = torch.cat([Jtr[:,:8], mano_verts[:,443].unsqueeze(1), Jtr[:,8:]], 1)
        Jtr = torch.cat([Jtr[:,:12], mano_verts[:,672].unsqueeze(1), Jtr[:,12:]], 1)
        Jtr = torch.cat([Jtr[:,:16], mano_verts[:,555].unsqueeze(1), Jtr[:,16:]], 1)
        Jtr = torch.cat([Jtr[:,:20], mano_verts[:,744].unsqueeze(1), Jtr[:,20:]], 1)
        
        # Jtr = torch.cat(Jtr, 2).permute(0,2,1)
        return Jtr
    
    def nimble_v_2_mano_j_reg(self, nimble_verts):#[b,5990,3]

        batch_size = nimble_verts.shape[0]
        
        # [b, 3, 5990] x [b, 5990, 21] -> [b, 3, 21]
        # self.jreg_mano: [21, 5990]
        Jtr = torch.matmul(self.jreg_mano.repeat(batch_size,1,1), nimble_verts)
        # Jtr = Jtr.permute(0, 2, 1) # b, 21, 3

        return Jtr

    def forward(self, hand_params, handle_collision=True):
        """
        Takes points in R^3 and first applies relevant pose and shape blend shapes.
        Then performs skinning.
        """
        batch_size = hand_params['pose_params'].shape[0]
        if self.use_pose_pca:
            full_pose = self.generate_full_pose(hand_params['pose_params'], normalized=True, with_root=True).view(-1, 20, 3)
        else:
            full_pose = hand_params['pose_params'].view(-1, 20, 3) # b, 20, 3

        th_v_shaped, jreg_joints = self.generate_hand_shape(hand_params['shape_params'],normalized=True, use_mean_shape=self.use_mean_shape)

        # if scale_gt is not None:
        #     mesh_v, bone_joints, rot = self.forward_full(th_v_shaped, full_pose, hand_params['trans'], jreg_joints, self.sw, self.pbs, scale_gt)
        # else: # pass estimated scale
            # mesh_v, bone_joints, rot = self.forward_full(th_v_shaped, full_pose, hand_params['trans'], jreg_joints, self.sw, self.pbs, hand_params['scale'])
        # ** no global scale and trans
        # root_trans = torch.zeros(jreg_joints.shape[0], 3).to(jreg_joints.device)
        # mesh_v, bone_joints, rot, center_joint = self.forward_full(th_v_shaped, full_pose, root_trans, jreg_joints, self.sw, self.pbs, None)
        mesh_v, bone_joints, rot = self.forward_full(points=th_v_shaped,      pose=full_pose,   root_trans=None, joints=jreg_joints, 
                                                     skinning_weight=self.sw, pose_bs=self.pbs, global_scale=None)
        
        skin_v = mesh_v[:, self.skin_v_sep:, :]

        if self.ifRender:
            # tex_img = self.generate_texture(hand_params['texture_params'], need=False)
            tex_img = self.generate_texture(hand_params['texture_params'])
            # create the texture object
            # texture = TexturesUV(
            #     maps=tex_img.permute(0, 3, 1, 2),  # Bx(3+3+3)xHxW
            # )
            tex_img_rgb = tex_img[:,:,:, :3].flip(dims=(3,))
            texture = Textures(
                maps=tex_img_rgb,  # BxHxWxC(3)
                verts_uvs=self.verts_uv.repeat(batch_size, 1, 1), 
                faces_uvs=self.faces_uv.repeat(batch_size, 1, 1), 
            )
        else:
            # not generate texture
            tex_img = self.generate_texture(hand_params['texture_params'], need=False)
            texture = None

        if handle_collision: # this is time-consuming
            skin_v = self.handle_collision(mesh_v)
            mesh_v[:, self.skin_v_sep:, :] = skin_v

        faces = self.skin_f.repeat(skin_v.shape[0], 1, 1)
        skin_p3dmesh = Meshes(skin_v, faces, texture)
        # skin_p3dmesh = smooth_mesh(skin_p3dmesh) # this is time-consuming
        del faces

        skin_mano_v = self.nimble_to_mano(skin_v, is_surface=True)
        joints = self.mano_v2j_reg(skin_mano_v)
        # joints = self.nimble_v_2_mano_j_reg(skin_v)

        # muscle_v = mesh_v[:,self.bone_v_sep:self.skin_v_sep,:]
        # bone_v = mesh_v[:,:self.bone_v_sep,:]

        return {
            'nimble_joints': bone_joints, # 25 joints
            'joints': joints, # Freihand joints, 21
            'verts': skin_v, # 5990 verts
            'faces': None, # faces, # very big number
            'rot': rot, # b, 3
            'skin_meshes': skin_p3dmesh, # meshes
            'mano_verts': skin_mano_v, # 5990 -> 778 verts according to mano
            'textures': tex_img,
        }


    def forward_full(self, points, pose, root_trans, joints, skinning_weight, pose_bs=None, global_scale=None):
        # pose: b, 20, 3
        batch_size = pose.shape[0]

        # Convert axis-angle representation to rotation matrix rep.
        th_pose_map, th_rot_map = th_posemap_axisang_2output(pose.view(batch_size, -1))
        th_full_pose = pose.view(batch_size, -1, 3) # b, 20, 3
        rot = th_full_pose[:, 0] # b, 3
        root_rot = batch_rodrigues(rot).view(batch_size, 3, 3) # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        th_j = joints

        if pose_bs is not None:
        # th_pose_map: 1, 19*3
        # points: B, N, 3
        # template_muscle_tet_pose_bs: N, 3, 25*3
        # with pose blend shape
            points_pose_bs = points + torch.matmul(
                pose_bs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        else:
            points_pose_bs = points

        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1) # b, 1, 3 -> b, 3 dims, 1
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2))) # b, 3, 3+1 -> b, 4, 4

        # Rotate each part
        for i in range(STATIC_JOINT_NUM - 1):
            i_val_joint = int(i + 1)
            if i_val_joint in JOINT_ID_BONE_DICT:
                i_val_bone = JOINT_ID_BONE_DICT[i_val_joint]
                joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone * 9].contiguous().view(batch_size, 3, 3)
            else:
                joint_rot = self.identity_rot.repeat(batch_size, 1, 1) # [1, 1, 1] -> [b, 3, 3]

            joint_j = th_j[:, i_val_joint, :].contiguous().view(batch_size, 3, 1)
            parent = self.kintree_parents[i_val_joint]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))

            th_results.append(torch.matmul(th_results[parent], joint_rel_transform)) # [b, 4, 4]

        th_results_global = th_results # 25 * [b, 4, 4]
        th_results2 = torch.zeros((batch_size, 4, 4, STATIC_JOINT_NUM),
                                  dtype=root_j.dtype,
                                  device=root_j.device) # [b, 4, 4, 25]

        for i in range(STATIC_JOINT_NUM):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i], # [b, 3]
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1) # cat: [b, 1] -> [b, 4]
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2)) # b, 4, 4 x b, 4, 1 => b, 4, 1
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)
        

        skinning_weight = skinning_weight.reshape(1, -1, STATIC_JOINT_NUM)
        th_verts = self.compute_warp(batch_size, points_pose_bs, skinning_weight, th_results2)

        # joints with pose
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3] # b, 25, 4, 4 -> b, 25, 3
        
        # https://github.com/reyuwei/NIMBLE_model/issues/5, 
        # NIMBLE model is in millimeter unit. Scale by 0.001 to get MANO scale (in meter unit).
        global_scale_meter = 0.001 * torch.ones(batch_size, 1).to(th_j.device)
        center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1) # [b, 1, 3]
        th_jtr = th_jtr - center_joint
        th_verts = th_verts - center_joint # [b, 14970, 3]

        verts_scale = global_scale_meter.expand(th_verts.shape[0], th_verts.shape[1])
        verts_scale = verts_scale.unsqueeze(2).repeat(1, 1, 3)
        th_verts = th_verts * verts_scale
        th_verts = th_verts + center_joint

        j_scale = global_scale_meter.expand(th_jtr.shape[0], th_jtr.shape[1])
        j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
        th_jtr = th_jtr * j_scale
        th_jtr = th_jtr + center_joint # x+(y-x)*s

        # global scaling
        if global_scale is not None:
            center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint

            verts_scale = global_scale.expand(th_verts.shape[0], th_verts.shape[1])
            verts_scale = verts_scale.unsqueeze(2).repeat(1, 1, 3)
            th_verts = th_verts * verts_scale
            th_verts = th_verts + center_joint

            j_scale = global_scale.expand(th_jtr.shape[0], th_jtr.shape[1])
            j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
            th_jtr = th_jtr * j_scale
            th_jtr = th_jtr + center_joint

        # global translation
        if root_trans is not None:
            root_position = root_trans.view(batch_size, 1, 3)
            center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1)
            offset = root_position - center_joint
        
            th_jtr = th_jtr + offset # origin-root + x
            th_verts = th_verts + offset

        return th_verts, th_jtr, rot


    def mesh_collision(self, floating_verts, floating_verts_normals, steady_verts, steady_faces):
        ### go to trimesh
        batch_size = floating_verts.shape[0]
        for i in range(batch_size):
            mesh_muscle = trimesh.Trimesh(steady_verts[i].detach().cpu().numpy(),
                                        steady_faces.squeeze().detach().cpu().numpy())
            skin_in_muscle = mesh_muscle.contains(floating_verts[i].detach().cpu().numpy())
            skin_surf_in_muscle = self.skin_v_surface_mask & torch.from_numpy(skin_in_muscle).to(self.device)
            if skin_surf_in_muscle.sum() <= 1:
                continue
            inside_verts = floating_verts[i][skin_surf_in_muscle].reshape(-1, 3)
            inside_verts_normal = floating_verts_normals[i][skin_surf_in_muscle].reshape(-1, 3)

            ## moving target using ray-triangle hit
            locations, index_ray, index_tri = mesh_muscle.ray.intersects_location(inside_verts.squeeze().detach().cpu().numpy(), 
                                            inside_verts_normal.squeeze().detach().cpu().numpy())
            locations = locations + 2* inside_verts_normal.squeeze().detach().cpu().numpy()[index_ray] # outside 2 mm
            index_ray = torch.from_numpy(index_ray).to(self.device)
            offset = torch.zeros_like(inside_verts)
            offset[index_ray] = torch.from_numpy(locations).float().to(self.device) - inside_verts[index_ray]

            ## move
            skin_v_offset = torch.zeros_like(floating_verts[i])
            skin_v_offset[skin_surf_in_muscle] = offset

            hard_result = floating_verts[i] + skin_v_offset
            soft_result = floating_verts[i] + (self.skin_v_node_weight.unsqueeze(-1) * skin_v_offset).sum(1)
            final_result = 0.7 * hard_result + 0.3 * soft_result
            floating_verts[i] = final_result
            
        return floating_verts


    def handle_collision(self, th_verts):
        muscle_v = th_verts[:,self.bone_v_sep:self.skin_v_sep,:]
        skin_v = th_verts[:,self.skin_v_sep:,:]
        interp_meshes_skin = Meshes(skin_v, self.skin_f.repeat(skin_v.shape[0], 1, 1))

        skin_v = self.mesh_collision(skin_v, interp_meshes_skin.verts_normals_padded(), muscle_v, self.muscle_f)
       
        return skin_v
        