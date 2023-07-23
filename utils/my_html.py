
import torch
from torch.autograd import Variable
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Textures
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
from utils.HTML_release_both_hands.utils.HTML import MANO_SMPL_HTML
fdir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(fdir)
MANO_dir = os.path.join(fdir,'mano')

class MyHTMLLayer(torch.nn.Module):
    def __init__(self, device, tex_ncomp=101):
        super(MyHTMLLayer, self).__init__()
        self.mesh_num = 778
        self.bases_num = 10 
        self.keypoints_num = 16
        self.device = device

        MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/MANO_RIGHT.pkl')
        # dd = pickle.load(open(MANO_file, 'rb'),encoding='latin1')
        # self.mesh_face = Variable(torch.from_numpy(np.expand_dims(dd['f'],0).astype(np.int16)).to(device=device))

        # self.mano_layer = ManoLayer(center_idx=9, flat_hand_mean=False, side='right', mano_root=MANO_dir,\
                        # use_pca = use_pose_pca, ncomps=pose_ncomp)

        tex_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils/HTML_release_both_hands/TextureBasis/model_sr/model.pkl")
        uv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils/HTML_release_both_hands/TextureBasis/uvs_right.pkl")
        self.html = MANO_SMPL_HTML(MANO_file, tex_path, uv_path)

        self.verts_uvs = torch.unsqueeze(self.html.verts_uvs, 0).cuda()
        self.faces_uvs= torch.unsqueeze(self.html.faces_uvs, 0).cuda()
        self.faces_idx = torch.unsqueeze(self.html.faces_idx, 0).cuda()


    def forward(self, tex_params, verts, faces):
        batch_size = tex_params['pose_params'].shape[0]
        verts_uvs = self.verts_uvs.repeat((batch_size, 1, 1)).contiguous()
        faces_idx = self.faces_idx.repeat((batch_size, 1, 1)).contiguous()
        faces_uvs = self.faces_uvs.repeat((batch_size, 1, 1)).contiguous()

        new_tex_img = self.html.get_mano_texture(tex_params)
        new_tex_img = new_tex_img.contiguous()
        tex = Textures(
                maps=new_tex_img,  # BxHxWxC(3)
                verts_uvs=verts_uvs, 
                faces_uvs=faces_uvs, 
            )
        #* not use this joints, but regress it later. Idk why.
        # mesh_face = self.mesh_face.repeat(batch_size, 1, 1)
        # skin_p3dmesh = Meshes(verts, mesh_face)
        skin_p3dmesh = Meshes(verts=verts, faces=faces_idx, textures=tex)
        return {
            # 'nimble_joints': bone_joints, # 25 joints
            # 'joints': joints, # mano joints, 21
            # 'verts': verts, # 5990 verts
            # 'faces': None, # faces, # very big number
            # 'rot': rot, # b, 3
            'skin_meshes': skin_p3dmesh, # smoothed verts and faces
            # 'mano_verts': verts, # 5990 -> 778 verts according to mano
            'textures': new_tex_img,
        }

