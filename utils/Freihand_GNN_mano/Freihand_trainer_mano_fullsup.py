# from graph_network import *
from mano_network import *
from dataloader.Freihand_dataloader_3d import *
from sys import platform
import os
import os.path as osp
import numpy as np
from argparse import ArgumentParser
import time
import copy
import torch.nn.functional as F
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5])



 ####loss 
def edge_length_loss(pred, gt, face, is_valid=None):

    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
    # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
    # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    diff1 = torch.abs(d1_out - d1_gt) #* valid_mask_1
    diff2 = torch.abs(d2_out - d2_gt) #* valid_mask_2
    diff3 = torch.abs(d3_out - d3_gt) #* valid_mask_3
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()   

# align for the test
def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


class average_metric():
    def __init__(self):
        self.num = 0.0
        self.value = 0.0 

    def update(self,metric):
        self.num += 1 
        self.value+= metric

    def parse(self):
        if self.num<1:
            return 0
        else:
            return self.value/self.num

    def reset(self):
        self.num = 0.0
        self.value = 0.0

def write_str(pstr,fpath):
    if not os.path.exists(fpath):
        mode = 'w'
    else:
        mode = 'a'
    with open(fpath,mode) as f:
        f.write(pstr)
    return
        

# this is the training files
class dense_pose_Trainer(object):
    def __init__(self, train_data_loader, test_data_loader,mano_dir = 'utils/Freihand_GNN_mano/template',save_dir='./results'):
        self.train_loader = train_data_loader
        self.test_loader = test_data_loader
        mano_path = os.path.join(mano_dir,'MANO_RIGHT.pkl')
        with open(mano_path,'rb') as f:
            self.mano_data = pickle.load(f, encoding='latin1')
        # set the log_path
        self.logt_path = os.path.join(save_dir,'log.txt')
        self.bmod_path = os.path.join(save_dir,'best_model.pth')
        self.tmod_path = os.path.join(save_dir,'tmp_model.pth')
        self.train_log = []
        self.test_log = []

    # this is the main file, contain both training and  testing
    def train(self, model, num_epochs):

        global_loss = 9999.9
        global_loss_vert = 9999.9
        lr = 1e-4#0.0001
        for epoch in range(1, num_epochs + 1):
            if epoch > 1 and epoch % 50 == 0:
                lr = lr * 0.1

            #optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(self.opt.beta1, 0.999))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.train_epoch(epoch, model, optimizer)
            vertex_loss, pose_loss = self.test(epoch, model)
            save_dict = {'model':model,'train':self.train_log,'test':self.test_log}
            if pose_loss < global_loss:
            # if (vertex_loss<global_loss_vert) or  (pose_loss < global_loss):
                global_loss = pose_loss
                global_loss_vert = vertex_loss
                print("3D hand pose:", global_loss, "3D Vert:",global_loss_vert )
                # save the model
                # torch.save(model,'/home/ziwei/Freihand_demo/results/model.pkl')
                # tmp_dir = os.path.split(__file__)[0]
                # msave_path = os.path.join(tmp_dir,'results/model.pkl')
                torch.save(save_dict, self.bmod_path)
            else:
                print("No update, 3D hand pose:",global_loss, "3D Vert:",global_loss_vert)
            # save the log (both file and print_str)            
            # generate print_str and update
            print_str = '%d-epoch:\n' %(epoch)
            print_str+= 'train:%.6f\n' % (self.train_log[-1])
            print_str+= 'test:%.6f\n' %(self.test_log[-1])
            print_str+= 'best results: %.6f epoch %d\n' %(min(self.test_log),self.test_log.index(min(self.test_log))+1)
            print_str+= '-'*50+'\n'
            write_str(pstr= print_str,fpath=self.logt_path)
            torch.save(save_dict,self.tmod_path)

    # from image projection to 3d
    def uvd2xyz(self, uvd, K):
        fx, fy, fu, fv = K[:, 0, 0].reshape(uvd.shape[0], 1), K[:, 1, 1].reshape(uvd.shape[0], 1), K[:, 0, 2].reshape(
            uvd.shape[0], 1), K[:, 1, 2].reshape(uvd.shape[0], 1)
        xyz = torch.zeros_like(uvd).cuda()
        xyz[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
        xyz[:, :, 1] = (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
        xyz[:, :, 2] = uvd[:, :, 2]
        return xyz

    # from 3d to image projection
    def xyz2uvd(self, xyz, K):
        fx, fy, fu, fv = K[:, 0, 0].reshape(xyz.shape[0], 1), K[:, 1, 1].reshape(xyz.shape[0], 1), K[:, 0, 2].reshape(
            xyz.shape[0], 1), K[:, 1, 2].reshape(xyz.shape[0], 1)
        uvd = torch.zeros_like(xyz).cuda()
        uvd[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
        uvd[:, :, 1] = (xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
        uvd[:, :, 2] = xyz[:, :, 2]
        return uvd

    # change the order of 16 keypoints and get 5 tips from mesh
    def get_keypoints_from_mesh_np(self, mesh_vertices, keypoints_regressed):
        """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
        kpId2vertices = {
            4: [744],  # ThumbT
            8: [320],  # IndexT
            12: [443],  # MiddleT
            16: [555],  # RingT
            20: [672]  # PinkT
        }
        keypoints = [0.0 for _ in range(21)]  # init empty list

        # fill keypoints which are regressed
        mapping = {0: 0,  # Wrist
                   1: 5, 2: 6, 3: 7,  # Index
                   4: 9, 5: 10, 6: 11,  # Middle
                   7: 17, 8: 18, 9: 19,  # Pinky
                   10: 13, 11: 14, 12: 15,  # Ring
                   13: 1, 14: 2, 15: 3}  # Thumb

        for manoId, myId in mapping.items():
            keypoints[myId] = keypoints_regressed[:,manoId, :]

        # get other keypoints from mesh
        for myId, meshId in kpId2vertices.items():
            keypoints[myId] = torch.mean(mesh_vertices[:,meshId, :], 1)

        #print(np.array(keypoints).shape,keypoints[0].shape )
        keypoints = torch.stack(keypoints)
        return keypoints

    # get keypoints from mesh, this is only true for freihand
    def xyz_from_vertice(self, vertice):
        self.Jreg = self.mano_data['J_regressor']
        np_J_regressor = torch.from_numpy(self.Jreg.toarray().T).float()
        #print(vertice.shape, np_J_regressor.shape)
        joint_x = torch.matmul(vertice[:, :, 0], np_J_regressor.cuda())
        joint_y = torch.matmul(vertice[:, :, 1], np_J_regressor.cuda())
        joint_z = torch.matmul(vertice[:, :, 2], np_J_regressor.cuda())
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
        coords_kp_xyz3 = self.get_keypoints_from_mesh_np(vertice, joints)
        #print(joints.shape,coords_kp_xyz3.shape)

        return coords_kp_xyz3

    # this is the function for training one epoch    
    def train_epoch(self, epoch, model, optimizer):
        train_log = average_metric()
        model.train()
        for step, (batch_image, batch_pose_3d, batch_vertices,others) in enumerate(self.train_loader):
            # if step>=10: 
            #     break
            batch_vertices_all = []
            batch_vertices_all.append(batch_vertices)#B 778 3;B 389 3;B 195 3;B 98 3
            # for i in range(len(down_transform_list)-1):
            #     verts = Pool(batch_vertices.cuda(), down_transform_list[i].cuda())
            #     batch_vertices_all.append(verts)
            # vertice_pred_list = model(batch_image.cuda())
            prediction = model(batch_image.cuda())
            vertice_pred_list = prediction['mesh']
            vertex_loss = torch.zeros(1).squeeze().cuda()
            edge_length_loss_right = torch.zeros(1).squeeze().cuda()
            pose3d_loss = torch.zeros(1).squeeze().cuda()
            lreg_beta = torch.zeros(1).squeeze()
            lreg_theta = torch.zeros(1).squeeze()
            joint_pred = self.xyz_from_vertice(vertice_pred_list[-1]).permute(1,0,2)
            blossfn = nn.L1Loss() #nn.SmoothL1Loss()#nn.L1Loss()
            pose3d_loss = blossfn(joint_pred, batch_pose_3d.cuda())
            # for idx in range(4):
            #     vertex_loss += blossfn(vertice_pred_list[3-idx],batch_vertices_all[idx].cuda())
            #     faces_right =tmp['face'][idx].astype(np.int16)
            #     edge_length_loss_right += edge_length_loss(vertice_pred_list[3-idx],batch_vertices_all[idx].cuda(),faces_right, is_valid=None)
            vertex_loss += blossfn(vertice_pred_list[0],batch_vertices_all[0].cuda())
            faces_right =tmp['face'][0].astype(np.int16)
            edge_length_loss_right += edge_length_loss(vertice_pred_list[0],batch_vertices_all[0].cuda(),faces_right, is_valid=None)            
            # compute the regulazation loss
            beta = prediction['beta']
            theta = prediction['theta']

            lreg_beta = beta.pow(2).sum(dim=-1).sqrt().mean()*10
            lreg_theta = theta.pow(2).sum(dim=-1).sqrt().mean()*10      

            total_loss = (vertex_loss*1000.0 + edge_length_loss_right*100.0 + pose3d_loss*1000.0)/20.0+ (lreg_beta+ lreg_theta)#*10
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if step % 10==0:
                print("Epoch", epoch, "|", "Step",step, "|",
                "Total_loss",total_loss.item(), "|","vertex loss", vertex_loss.item()*1000.0/20.0,
                "|","edg loss", edge_length_loss_right.item()*100.0/20.0,"|","pose loss", pose3d_loss.item()*1000.0/20.0,
                "|", 'Lreg_beta',lreg_beta.item(),"|",'Lreg_theta',lreg_theta.item())
            # update train_loss
            train_log.update(total_loss.item())

        self.train_log.append(train_log.parse())


    def test(self, epoch, model):
        model.eval()
        with torch.no_grad():
            vert_3d_all = []
            pose_3d_all = []
            vert_3d_all_gt =[]
            pose_3d_all_gt = []
            pose_align_all = []
            vert_align_all = []
            for step, (batch_image, batch_pose3d, batch_vertice,others) in enumerate(self.test_loader):
                
                # vertice_pred_list = model(batch_image.cuda())
                prediction = model(batch_image.cuda())
                vertice_pred_list = prediction['mesh']
                #normalized pred
                vertice_pred = vertice_pred_list[-1]
                joints_pred = self.xyz_from_vertice(vertice_pred).permute(1,0,2) # b, 21, 3 -> 21, b, 3

                vert_3d_all.append(vertice_pred.detach().cpu().numpy())
                pose_3d_all.append(joints_pred.detach().cpu().numpy()) # [tensor(21,b,3), tensor, ...]
                vert_3d_all_gt.append(batch_vertice.detach().cpu().numpy())
                pose_3d_all_gt.append(batch_pose3d.detach().cpu().numpy())
            vert_3d = (np.concatenate(vert_3d_all, axis=0))
            pose_3d = (np.concatenate(pose_3d_all, axis=0))
            vert_3d_gt = (np.concatenate(vert_3d_all_gt, axis=0))
            pose_3d_gt = (np.concatenate(pose_3d_all_gt, axis=0))

            for idx in range(vert_3d.shape[0]): # 0 - 3000+
                #align prediction
                pose_pred_aligned=align_w_scale(pose_3d_gt[idx], pose_3d[idx])
                vert_pred_aligned=align_w_scale(vert_3d_gt[idx], vert_3d[idx])
                pose_align_all.append(pose_pred_aligned)
                vert_align_all.append(vert_pred_aligned)
            vert_align_all = torch.from_numpy(np.array(vert_align_all)).cuda()
            pose_align_all = torch.from_numpy(np.array(pose_align_all)).cuda()
            vert_3d_gt = torch.from_numpy(vert_3d_gt).cuda()
            pose_3d_gt = torch.from_numpy(pose_3d_gt).cuda()

            vert_3d_loss = torch.linalg.norm((vert_align_all - vert_3d_gt), ord=2,dim=-1)
            pose_3d_loss = torch.linalg.norm((pose_align_all - pose_3d_gt), ord=2,dim=-1)
            vert_3d_loss = (np.concatenate(vert_3d_loss.detach().cpu().numpy(),axis=0)).mean()
            pose_3d_loss = (np.concatenate(pose_3d_loss.detach().cpu().numpy(),axis=0)).mean()

            print("Evaluation vert_3d:",vert_3d_loss,"|","pose 3d:", pose_3d_loss)
            self.test_log.append(vert_3d_loss.item())
            return vert_3d_loss, pose_3d_loss

if __name__ == '__main__':
    # path = "/mnt/data/ziwei/Freihand/"
    path = '/mnt/data/allusers/haipeng/HandData/FreiHAND'
    # path = '/code/Hand/HandDset/FreiHAND'
    batch_size = 200#320#200#128#128*8
    epochs = 200
    dataset = FreiHAND(root = path, mode="training")
    trainloader_synthesis = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    
    dataset_test = FreiHAND_test(root = path, mode="evaluation")
    test_synthesis = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=10)

    # work_dir = "/home/ziwei/Freihand_demo"
    # work_dir = '/mnt/data/allusers/haipeng/HandMethods/Freihand_GNNdemo'
    work_dir = os.path.split(os.path.abspath(__file__))[0]
    template_fp = os.path.join(work_dir, "template", "template.ply")
    transform_fp = os.path.join(work_dir, "template", "transform.pkl")

    seq_length = [27, 27, 27, 27] #the length of neighbours
    dilation = [1, 1, 1, 1] # whether dilate sample
    ds_factors = [2, 2, 2, 2] #downsample factors 2, 2, 2, 2 

   
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation)
    model = YTBHand(spiral_indices_list, up_transform_list)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    mano_dir = os.path.join(work_dir,'mano')
    save_dir = os.path.join(work_dir,'results')
    if not os.path.exists(save_dir):
        save_dir = os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'mano_L1')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trainer = dense_pose_Trainer(trainloader_synthesis, test_synthesis,mano_dir=mano_dir,save_dir=save_dir)
    trainer.train(model, epochs)




            

