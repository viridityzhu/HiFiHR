
import torch
from torch.autograd import Variable
from pytorch3d.structures.meshes import Meshes
import pickle
import numpy as np
import os
import sys

from torch.nn import Module

from utils.mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from utils.manopth import rodrigues_layer, rotproj, rot6d
from utils.manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)

from utils.hand_3d_model import rodrigues, get_poseweights
fdir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(fdir)
MANO_dir = os.path.join(fdir,'mano')

class MyMANOLayer(torch.nn.Module):
    def __init__(self, ifRender, device, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, use_pose_pca=True):
        super(MyMANOLayer, self).__init__()
        self.pose_num = pose_ncomp
        self.mesh_num = 778
        self.bases_num = 10 
        self.keypoints_num = 16
        self.device = device

        MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/MANO_RIGHT.pkl')
        dd = pickle.load(open(MANO_file, 'rb'),encoding='latin1')
        self.mesh_face = Variable(torch.from_numpy(np.expand_dims(dd['f'],0).astype(np.int16)).to(device=device))

        self.mano_layer = ManoLayer(center_idx=9, flat_hand_mean=False, side='right', mano_root=MANO_dir,\
                        use_pca = use_pose_pca, ncomps=pose_ncomp)


    def forward(self, hand_params, handle_collision=True):
        batch_size = hand_params['pose_params'].shape[0]
        verts, _ = self.mano_layer(hand_params['pose_params'], hand_params['shape_params']) # bs*778*3
        #* not use this joints, but regress it later. Idk why.
        mesh_face = self.mesh_face.repeat(batch_size, 1, 1)
        skin_p3dmesh = Meshes(verts, mesh_face)
        return {
            # 'nimble_joints': bone_joints, # 25 joints
            # 'joints': joints, # mano joints, 21
            # 'verts': verts, # 5990 verts
            # 'faces': None, # faces, # very big number
            # 'rot': rot, # b, 3
            'skin_meshes': skin_p3dmesh, # smoothed verts and faces
            'mano_verts': verts, # 5990 -> 778 verts according to mano
            # 'textures': tex_img,
        }

# !depreciated
class MyMANOLayer_de(torch.nn.Module):
    def __init__(self, ifRender, device, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, use_pose_pca=True):
        super(MyMANOLayer_de, self).__init__()
        MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/MANO_RIGHT.pkl')
        dd = pickle.load(open(MANO_file, 'rb'),encoding='latin1')
        self.kintree_table = dd['kintree_table']
        self.id_to_col = {self.kintree_table[1,i] : i for i in range(self.kintree_table.shape[1])} 
        self.parent = {i : self.id_to_col[self.kintree_table[0,i]] for i in range(1, self.kintree_table.shape[1])}  

        self.pose_num = pose_ncomp
        self.mesh_num = 778
        self.bases_num = 10 
        self.keypoints_num = 16
        self.device = device

        self.mesh_mu = Variable(torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).to(device=device)) # zero mean
        self.mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).to(device=device))
        self.posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).to(device=device))
        self.J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).to(device=device))
        self.weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).to(device=device))
        self.hands_components = Variable(torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:self.pose_num]), 0).astype(np.float32)).to(device=device))
        self.hands_mean = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).to(device=device))
        self.mesh_face = Variable(torch.from_numpy(np.expand_dims(dd['f'],0).astype(np.int16)).to(device=device))


    def forward(self, hand_params, handle_collision=True):
        jv, faces, tsa_poses = self.rot_pose_beta_to_mesh(hand_params['rot'], hand_params['pose_params'], hand_params['shape_params'])#rotation pose shape
        # jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
        joints = jv[:,0:21]
        verts = jv[:,21:]
        skin_p3dmesh = Meshes(verts, faces)
        return {
            # 'nimble_joints': bone_joints, # 25 joints
            'joints': joints, # mano joints, 21
            # 'verts': verts, # 5990 verts
            # 'faces': None, # faces, # very big number
            # 'rot': rot, # b, 3
            'skin_meshes': skin_p3dmesh, # smoothed verts and faces
            'mano_verts': verts, # 5990 -> 778 verts according to mano
            # 'textures': tex_img,
        }
    def rot_pose_beta_to_mesh(self, rots, poses, betas):
        '''
            Using MANO, convert the provided rotation, theta (pose), and beta (shape) into mesh joints, verts, faces, and poses.
        '''
        root_rot = rots.unsqueeze(1)

        batch_size = rots.size(0)   

        mesh_face = self.mesh_face.repeat(batch_size, 1, 1)
        # [b,15,3] [0:3]index [3:6]mid [6:9]pinky [9:12]ring [12:15]thumb


        # for visualization
        #rots = torch.zeros_like(rots); rots[:,0]=np.pi/2


        # 1. pose = mean + aa
        # 网络预测的axis angle + hand_mean (rest pose), 
        # 再加上 root_rot(wrist的位置，代码里面写为 (0,0,0), 是相对位置, 构造root-relative的结果). [bs, 16, 3]

        #poses = torch.ones_like(poses)*1
        #poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)   
        poses = (self.hands_mean + torch.matmul(poses.unsqueeze(1), self.hands_components).squeeze(1)).view(batch_size,self.keypoints_num-1,3)
        # poses = torch.cat((root_rot.repeat(batch_size,1).view(batch_size,1,3),poses),1) # [b,16,3]
        poses = torch.cat((root_rot,poses),1) # [b,16,3]

        # 2. shape: rest + blend
        # rest + Blend shape
        v_shaped =  (torch.matmul(betas.unsqueeze(1), 
                    self.mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,self.bases_num,-1)).squeeze(1)    
                    + self.mesh_mu.repeat(batch_size,1,1).view(batch_size, -1)).view(batch_size, self.mesh_num, 3)      
        
        # rest + Blend shape + blend pose
        pose_weights = get_poseweights(poses, batch_size)#[b,135]   
        v_posed = v_shaped + torch.matmul(self.posedirs.repeat(batch_size,1,1,1),
                (pose_weights.view(batch_size,1,(self.keypoints_num - 1)*9,1)).repeat(1,self.mesh_num,1,1)).squeeze(3)
        # Final T pose with transformation done !

        # 3. regress joints from verts
        # rest verts -> joints
        J_posed = torch.matmul(v_shaped.permute(0,2,1), self.J_regressor.repeat(batch_size,1,1).permute(0,2,1))
        J_posed = J_posed.permute(0, 2, 1)
        J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]
            
        pose = poses.permute(1, 0, 2)
        pose_split = torch.split(pose, 1, 0)

        # 4. rotate the joints aa
        angle_matrix =[]
        for i in range(self.keypoints_num):
            out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
            angle_matrix.append(out)

        #with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)
        with_zeros = lambda x:\
            torch.cat((x, Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1).to(device=self.device))  ),1)

        pack = lambda x: torch.cat((Variable(torch.zeros(batch_size,4,3).to(device=self.device)),x),2) 


        results = {}
        results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size,3,1)),2))

        for i in range(1, self.kintree_table.shape[1]):
            tmp = with_zeros(torch.cat((angle_matrix[i],
                            (J_posed_split[i] - J_posed_split[self.parent[i]]).view(batch_size,3,1)),2)) 
            results[i] = torch.matmul(results[self.parent[i]], tmp)

        # 16个手指节点的 3D xyz. [(bs ,4, 4), (bs, 4, 4), … , (bs ,4, 4)].
        results_global = results

        results2 = []
            
        for i in range(len(results)):
            vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size,1).to(device=self.device)) ),1)).view(batch_size,4,1)
            results2.append((results[i]-pack(torch.matmul(results[i], vec))).unsqueeze(0))    

        results = torch.cat(results2, 0)
        
        T = torch.matmul(results.permute(1,2,3,0), self.weights.repeat(batch_size,1,1).permute(0,2,1).unsqueeze(1).repeat(1,4,1,1))
        Ts = torch.split(T, 1, 2)
        rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size,self.mesh_num,1).to(device=self.device)) ), 2)  
        rest_shape_hs = torch.split(rest_shape_h, 1, 2)

        # 经过蒙皮处理后的最终变形节点
        v = Ts[0].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[1].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[2].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[3].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, self.mesh_num)
    
        #v = v.permute(0,2,1)[:,:,:3] 
        # v: b, 4, 778 mesh_num
        # Rots = rodrigues(rots)[0]
        Jtr = []

        for j_id in range(len(results_global)):
            Jtr.append(results_global[j_id][:,:3,3:4]) # b, 3, 1

        # Add finger tips from mesh to joint list    
        '''
        Jtr.insert(4,v[:,:3,333].unsqueeze(2))
        Jtr.insert(8,v[:,:3,444].unsqueeze(2))
        Jtr.insert(12,v[:,:3,672].unsqueeze(2))
        Jtr.insert(16,v[:,:3,555].unsqueeze(2))
        Jtr.insert(20,v[:,:3,745].unsqueeze(2)) 
        '''
        # v is of shape: b, 3 (or more) dims, 778 samples
        # For FreiHand: add 5 joints.
        Jtr.insert(4,v[:,:3,320].unsqueeze(2))
        Jtr.insert(8,v[:,:3,443].unsqueeze(2))
        Jtr.insert(12,v[:,:3,672].unsqueeze(2))
        Jtr.insert(16,v[:,:3,555].unsqueeze(2))
        Jtr.insert(20,v[:,:3,744].unsqueeze(2))      
        
        Jtr = torch.cat(Jtr, 2) #.permute(0,2,1)
        
        # 再旋转 (根据root节点的旋转角)
        # v = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
        # Jtr = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)
        v = v[:,:3,:].permute(0,2,1)
        Jtr = Jtr.permute(0,2,1)
        
        #return torch.cat((Jtr,v), 1)
        return torch.cat((Jtr,v), 1), mesh_face, poses



class ManoLayer(Module):
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx', 'joint_rot_mode'
    ]

    def __init__(self,
                 center_idx=None,
                 flat_hand_mean=True,
                 ncomps=6,
                 side='right',
                 mano_root='mano/models',
                 use_pca=True,
                 root_rot_mode='axisang',
                 joint_rot_mode='axisang',
                 robust_rot=False):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()

        self.center_idx = center_idx
        self.robust_rot = robust_rot
        if root_rot_mode == 'axisang':
            self.rot = 3
        else:
            self.rot = 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components']

        self.smpl_data = smpl_data

        self.register_buffer('th_betas', torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs', torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs', torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer( 'th_v_template', torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer( 'th_J_regressor', torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]
                              ) if flat_hand_mean else smpl_data['hands_mean']
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Save as axis-angle
            self.register_buffer('th_hands_mean', th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer('th_comps', torch.Tensor(hands_components))
            self.register_buffer('th_selected_comps',
                                 torch.Tensor(selected_components))
        else:
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(
                th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer('th_hands_mean_rotmat', th_hands_mean_rotmat)

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self,
                th_pose_coeffs,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1),
                root_palm=torch.Tensor([0]),
                share_betas=torch.Tensor([0]),
                ):
        """
        Args:
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """
        # if len(th_pose_coeffs) == 0:
        #     return th_pose_coeffs.new_empty(0), th_pose_coeffs.new_empty(0)

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Remove global rot coeffs
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                 self.ncomps]
            if self.use_pca:
                # PCA components --> axis angles
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            # Concatenate back global rot
            th_full_pose = torch.cat([
                th_pose_coeffs[:, :self.rot],
                self.th_hands_mean + th_full_hand_pose
            ], 1)
            if self.root_rot_mode == 'axisang':
                # compute rotation matrixes from axis-angle while skipping global rotation
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, (
                'When not self.use_pca, '
                'th_pose_coeffs should have 4 dims, got {}'.format(
                    th_pose_coeffs.dim()))
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_pose_coeffs.shape[2:4]))
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        # Full axis angle representation with root joint
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       self.th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)

        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]] # 5, 1, 2, 4, 3
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, # 5 small
                                1, 2, 3, 17, # 1 thumb
                                4, 5, 6, 18, # 2 index
                                10, 11, 12, 19,  # 4 ring
                                7, 8, 9, 20]] # 3 middle

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        # # Scale to milimeters why??
        # th_verts = th_verts * 1000
        # th_jtr = th_jtr * 1000
        return th_verts, th_jtr
