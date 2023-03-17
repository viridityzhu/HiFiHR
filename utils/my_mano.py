
import torch
from pytorch3d.structures.meshes import Meshes

from utils.hand_3d_model import rot_pose_beta_to_mesh

class MyMANOLayer(torch.nn.Module):
    def __init__(self, ifRender, device, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, pm_dict_name="utils/NIMBLE_model/assets/NIMBLE_DICT_9137.pkl", tex_dict_name="utils/NIMBLE_model/assets/NIMBLE_TEX_DICT.pkl", nimble_mano_vreg_name="utils/NIMBLE_model/assets/NIMBLE_MANO_VREG.pkl", use_pose_pca=True):
        super(MyMANOLayer, self).__init__()

    def forward(self, hand_params, handle_collision=True):
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(hand_params['rot'], hand_params['pose_params'], hand_params['shape_params'])#rotation pose shape
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