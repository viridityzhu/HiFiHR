
import torch
from torch.autograd import Variable
from pytorch3d.structures.meshes import Meshes
import pickle
import numpy as np
import os

from utils.hand_3d_model import rodrigues, get_poseweights

class MyMANOLayer(torch.nn.Module):
    def __init__(self, ifRender, device, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, use_pose_pca=True):
        super(MyMANOLayer, self).__init__()
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
