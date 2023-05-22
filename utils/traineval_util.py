import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.optim as optim

import os
import matplotlib.pyplot as plt
import numpy as np


from utils.hand_3d_model import rot_pose_beta_to_mesh
from utils.fh_utils import Mano2Frei, RHD2Frei, HO3D2Frei, Frei2HO3D, AverageMeter
from utils.visualize_util import draw_2d_error_curve
import utils.visualize_util as visualize_util
from utils.fh_utils import proj_func
from utils.losses_util import bone_direction_loss, tsa_pose_loss#image_l1_loss, iou_loss, ChamferLoss,

import time
import json

def data_dic(data_batch, dat_name, set_name, args) -> dict:
    example_torch = {}
    if dat_name == 'FreiHand':
        # raw data
        if "trans_images" in data_batch:
            imgs = (data_batch['trans_images']).cuda()#[b,3,224,224]
        elif "images" in data_batch:
            imgs = (data_batch['images']).cuda()
        else:
            import pdb; pdb.set_trace()
        example_torch['imgs'] = imgs

        # refhand image
        if "refhand" in data_batch:
            example_torch['refhand'] = data_batch["refhand"]

        if "trans_Ks" in data_batch:
            Ks = data_batch['trans_Ks'].cuda()#[b,3,3]
        elif "Ks" in data_batch:
            Ks = data_batch['Ks'].cuda()#[b,3,3]
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'scales' in data_batch.keys():
            example_torch['scales'] = data_batch['scales'].float()#[b]
        
        #j2d_gt, masks, maskRGBs, manos, verts, joints, open_2dj, CRFmasks, CRFmaskRGBs = None, None, None, None, None, None, None, None, None
        example_torch['idxs'] = data_batch['idxs'].cuda()
        
        if 'CRFmasks' in data_batch.keys():
            CRFmasks = data_batch['CRFmasks'].cuda()
            example_torch['CRFmasks'] = CRFmasks
            example_torch['CRFmaskRGBs'] = imgs.mul(CRFmasks[:,2].unsqueeze(1).repeat(1,3,1,1).float())#
        if 'open_2dj' in data_batch.keys() or 'trans_open_2dj' in data_batch.keys():
            if 'trans_open_2dj' in data_batch.keys():
                example_torch['open_2dj'] = data_batch['trans_open_2dj'].cuda()
            elif 'open_2dj' in data_batch.keys():
                example_torch['open_2dj'] = data_batch['open_2dj'].cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!
            texture_idx_con = ((data_batch['idxs']<32560).float()+0.1).cuda()
            texture_con = torch.mean((torch.min(open_2dj_con.squeeze(),1)[0]>0.1).float().unsqueeze(1).mul(open_2dj_con.squeeze()),1)#[b]
            texture_con = torch.mul(texture_con,texture_idx_con)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con
            #open_2dj_con = (torch.min(open_2dj_con.squeeze(),1)[0]>0).float().unsqueeze(1).mul(open_2dj_con.squeeze()).unsqueeze(-1)
            '''
            idxs_con = ((idxs<32560).float()+1)/2
            texture_con = idxs_con.mul(texture_con)
            '''
        if 'manos' in data_batch.keys():
            manos = torch.squeeze(data_batch['manos'],1).cuda()#[b,61]
            example_torch['manos'] = manos
        if 'joints' in data_batch.keys():
            joints = data_batch['joints'].cuda()#[b,21,3]
            example_torch['joints'] = joints
            j2d_gt = proj_func(joints, Ks)
            example_torch['j2d_gt'] = j2d_gt
        if 'verts' in data_batch.keys():
            verts = data_batch['verts'].cuda()
            example_torch['verts'] = verts
        if 'masks' in data_batch.keys():
            masks = data_batch['masks'].cuda()#[b,3,224,224]
            example_torch['masks'] = masks
            #maskRGBs = data_batch['maskRGBs'].cuda()#[b,3,224,224]
            segms_gt = masks[:,0].long()#[b, 224, 224]# mask_gt
            example_torch['segms_gt'] = segms_gt
        if 'training' in set_name:
            
            if 'trans_masks' in data_batch.keys():
                masks = data_batch['trans_masks'].cuda()#[b,3,224,224]
                example_torch['masks'] = masks
                #maskRGBs = data_batch['maskRGBs'].cuda()#[b,3,224,224]
                segms_gt = masks[:,0].long()#[b, 224, 224]# mask_gt
                example_torch['segms_gt'] = segms_gt
            
            if 'trans_joints' in data_batch.keys():
                joints = data_batch['trans_joints'].cuda()#[b,21,3]
                example_torch['joints'] = joints
                j2d_gt = proj_func(joints, Ks)
                example_torch['j2d_gt'] = j2d_gt
            if "trans_verts" in data_batch.keys():
                verts = data_batch['trans_verts'].cuda()
                example_torch['verts'] = verts
            if args.semi_ratio is not None and 'j2d_gt' in example_torch and 'open_2dj' in example_torch:
                raw_idx = example_torch['idxs']%32560
                mix_open_2dj = torch.where((raw_idx<32560*args.semi_ratio).view(-1,1,1),example_torch['j2d_gt'],example_torch['open_2dj'])
                mix_open_2dj_con = torch.where((raw_idx<32560*args.semi_ratio).view(-1,1,1),torch.ones_like(example_torch['open_2dj_con']).to(device),example_torch['open_2dj_con'])
                example_torch['open_2dj'] = mix_open_2dj
                example_torch['open_2dj_con'] = mix_open_2dj_con
                
    elif dat_name == 'HO3D0':
        example_torch['imgs'] = (data_batch['img_crop']).cuda()#[b,3,224,224]
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        # only for HO3D
        Ks = Ks.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())# check

        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch.keys():
            j2d_gt = data_batch['uv21_crop'].cuda()
            j2d_gt = HO3D2Frei(j2d_gt)
            example_torch['j2d_gt'] = j2d_gt
        if 'xyz21' in data_batch.keys():
            joints = data_batch['xyz21'].cuda()#[b,21,3]
            joints = HO3D2Frei(joints)
            # only for HO3D
            joints = joints.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            example_torch['joints'] = joints
            j2d_gt_proj = proj_func(joints, Ks)
            '''
            
            Ks_raw = data_batch['camMat'].cuda()
            Ks_raw = Ks_raw.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            j2d_gt_raw = proj_func(joints, Ks_raw)
            crop_center = data_batch['crop_center0']
            image_center = torch.tensor([320,240]).float().unsqueeze(0).repeat(crop_center.shape[0],1)
            trans_image = image_center - crop_center
            scale = (Ks_raw[:,0,0]+Ks_raw[:,1,1])/2 torch.mean(joints,1)[:,2]
            trans_xy = trans_image.cuda()*torch.mean(joints,1)[:,2].unsqueeze(-1)*torch.reciprocal((Ks_raw[:,0,0]+Ks_raw[:,1,1])/2).unsqueeze(-1)
            joints_trans = joints + torch.cat((trans_xy, torch.zeros([trans_xy.shape[0],1]).to(device)),1).unsqueeze(1)
            proj_func(joints_trans, Ks_raw)
            #example_torch['j2d_gt'] = j2d_gt
            '''
        if 'open_2dj_crop' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!q
            texture_con = torch.mean(open_2dj_con.squeeze(),1)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con
    elif dat_name == 'HO3D':
        example_torch['imgs'] = torch_f.interpolate(data_batch['img_crop'],(224,224)).cuda()#[b,3,640,640] --> [b,3,224,224]
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        # only for HO3D
        Ks = Ks.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())# check
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch.keys():
            j2d_gt = data_batch['uv21_crop'].float().cuda()
            j2d_gt = HO3D2Frei(j2d_gt)
            example_torch['j2d_gt'] = j2d_gt
        if 'xyz21' in data_batch.keys():
            joints = data_batch['xyz21'].cuda()#[b,21,3]
            joints = HO3D2Frei(joints)
            # only for HO3D
            joints = joints.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            example_torch['joints'] = joints
            #j2d_gt_proj = proj_func(joints, Ks)
            #example_torch['j2d_gt'] = j2d_gt_proj
            
            #joints0 = data_batch['xyz210'].cuda()#[b,21,3]
            #joints0 = HO3D2Frei(joints0)
            # only for HO3D
            #joints0 = joints0.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            #j2d_gt_proj0 = proj_func(joints0, Ks)
            #example_torch['joints'] = joints0
            #example_torch['j2d_gt'] = j2d_gt_proj0
        if 'open_2dj_crop' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].float().cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!q
            texture_con = torch.mean(open_2dj_con.squeeze(),1)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con

    elif dat_name == 'RHD':
        '''
        imgs = data_batch['images'].cuda()#[b,3,320,320]
        Ks = data_batch['Ks'].cuda()
        uv21 = data_batch['uv21'].cuda()
        '''
        # for croped
        imgs = data_batch['img_crop'].cuda()#[b,3,224,224]
        example_torch['imgs'] = imgs
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch:
            j2d_gt = data_batch['uv21_crop'].cuda()#[B, 21, 2]
            j2d_gt = RHD2Frei(j2d_gt)
            example_torch['j2d_gt']=j2d_gt
        if 'xyz21' in data_batch:
            joints = data_batch['xyz21'].cuda()#[B, 21, 3]
            joints = RHD2Frei(joints)
            example_torch['joints']=joints
        '''
        if 'xyz_trans' in data_batch and 'K_crop_trans' in data_batch:
            joints_trans = data_batch['xyz_trans'].cuda()#[B, 21, 3]
            joints_trans = RHD2Frei(joints_trans)
            example_torch['joints_trans']=joints_trans
            j2d_gt_trans = proj_func(joints_trans, data_batch['K_crop_trans'].cuda())
        '''
        # Calculate projected 2D joints
        #j2d_syngt = proj_func(joints, Ks)
        if 'keypoint_scale' in data_batch:
            keypoint_scale = data_batch['keypoint_scale'].cuda()#[B]
            example_torch['keypoint_scale'] = keypoint_scale
        if "uv_vis" in data_batch:
            uv_vis = data_batch['uv_vis']#[B,21] True False
            uv_vis = RHD2Frei(uv_vis)
            example_torch['uv_vis'] = uv_vis
        '''
        if 'mask_crop' in data_batch:
            masks = data_batch['mask_crop'].cuda()#[B,1,224,224] 0 1
            masks = masks.repeat(1,3,1,1).float()#[B,3,224,224] 0 1
            maskRGBs = imgs.mul(masks)#
        '''
        if 'sides' in data_batch:
            # sides before flip to right
            side = data_batch['sides'].cuda()#[8,1]  0 left; 1 right
        
        if 'open_2dj' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].cuda()# idx(openpose) == idx(freihand)
            example_torch['open_2dj_con'] = data_batch['open_2dj_con'].cuda()
    
    elif dat_name == 'Dart':
        example_torch['imgs'] = data_batch['image'].cuda()
        example_torch['ortho_intr'] = data_batch['ortho_intr'].cuda()
        example_torch['Ps'] = None
        example_torch['Ks'] = None
        example_torch['scales'] = None
        
        example_torch['idxs'] = data_batch['idxs'].cuda()
        if 'mano_pose' in data_batch.keys():
            manos = torch.squeeze(data_batch['mano_pose'],1).cuda()#[b,61]
            example_torch['manos'] = manos
        if 'joints_3d' in data_batch.keys():
            joints = data_batch['joints_3d'].cuda()#[b,21,3]
            example_torch['joints'] = joints
            j2d_gt = data_batch['joints_2d'].cuda()
            example_torch['j2d_gt'] = j2d_gt
        if 'verts_3d' in data_batch.keys():
            verts = data_batch['verts_3d'].cuda()
            example_torch['verts'] = verts
        if 'image_mask' in data_batch.keys():
            masks = data_batch['image_mask'].cuda()#[b,3,224,224]
            example_torch['masks'] = masks
            #maskRGBs = data_batch['maskRGBs'].cuda()#[b,3,224,224]
            segms_gt = masks[:,0].long()#[b, 224, 224]# mask_gt
            example_torch['segms_gt'] = segms_gt
    
    return example_torch
        



def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)



def trans_proj(outputs, Ks_this, dat_name, is_ortho=False):
    if dat_name == 'FreiHand':
        output_joints = Mano2Frei(outputs['joints'])
        outputs['joints'] = output_joints
    elif dat_name == 'RHD':
        output_joints = Mano2Frei(outputs['joints'])
        outputs['joints'] = output_joints
    elif dat_name == 'HO3D':
        output_joints = Mano2Frei(outputs['joints'])
        outputs['joints'] = output_joints
    
    if 'joint_2d' in outputs:
        outputs['j2d'] = Mano2Frei(outputs['joint_2d'])
    if 'j2d' not in outputs:
        if is_ortho:
            proj_joints = orthographic_proj_withz(outputs['joints'], outputs['trans'], outputs['scale'])
            outputs['j2d'] = proj_joints[:, :, :2]
        else:    
            outputs['j2d'] = proj_func(output_joints, Ks_this)

    xyz_pred_list, verts_pred_list = [], []
    for i in range(outputs['joints'].shape[0]):
        if dat_name == "FreiHand":
            xyz_pred_list.append(outputs['joints'][i].cpu().detach().numpy())
        elif dat_name == "HO3D":
            output_joints_ho3d = Frei2HO3D(outputs['joints'])
            output_joints_ho3d = output_joints_ho3d.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            xyz_pred_list.append(output_joints_ho3d[i].cpu().detach().numpy())
        if 'vertices' in outputs:
            verts_pred_list.append(outputs['vertices'][i].cpu().detach().numpy())
    
    return outputs, xyz_pred_list, verts_pred_list


def trans_proj_j2d(outputs, Ks_this, scales=None, is_ortho=False, root_xyz=None, which_joints='joints'):
    j3d = outputs[which_joints]
    if root_xyz is not None and scales is not None:
        cal_scale = torch.norm(outputs['joints'][:, 9] - outputs['joints'][:, 10], dim=-1) # metric length of a reference bone
        scales = scales.to(j3d.device) / cal_scale
        scales = scales.unsqueeze(1).expand(j3d.shape[0], j3d.shape[1]).unsqueeze(2).repeat(1,1,3)
        j3d = j3d * scales
        j3d = j3d + root_xyz# recover the camera view coord
    if is_ortho:
        proj_joints = orthographic_proj_withz(j3d, outputs['trans'], outputs['scale'])
        j2d = proj_joints[:, :, :2]
    else:    
        j2d = proj_func(j3d, Ks_this)
    
    return j2d

# Dart ortho proj func
def ortho_project(points3d, ortho_cam):
    x, y = points3d[:, :, 0], points3d[:, :, 1]
    
    u = ortho_cam[:, 0].unsqueeze(1) * x + ortho_cam[:, 1].unsqueeze(1)
    v = ortho_cam[:, 0].unsqueeze(1) * y + ortho_cam[:, 2].unsqueeze(1)
    
    u = u.cpu().detach().numpy()
    v = v.cpu().detach().numpy()
    
    u_, v_ = u[:, np.newaxis], v[:, np.newaxis]
    
    proj_points = np.concatenate([u_, v_], axis=1) #[b, 2, 21]
    return np.transpose(proj_points, (0, 2, 1)) #[b, 21, 2]

def save_2d_result(j2d_pred_ED_list,j2d_proj_ED_list,j2d_detect_ED_list,args,j2d_pred_list=[], j2d_proj_list=[], j2d_gt_list=[], j2d_detect_list=[], j2d_detect_con_list=[], epoch=0):
    save_dir = os.path.join(args.base_output_dir,'joint2d_result',str(epoch))
    os.makedirs(save_dir, exist_ok=True)

    j2d_pred_ED = np.asarray(j2d_pred_ED_list)
    j2d_proj_ED = np.asarray(j2d_proj_ED_list)
    j2d_detect_ED = np.asarray(j2d_detect_ED_list)
    print("Prediction - Per Joint Mean Error:",np.mean(j2d_pred_ED,0))
    print("Projection - Per Joint Mean Error:",np.mean(j2d_proj_ED,0))
    print("Detection - Per Joint Mean Error:",np.mean(j2d_detect_ED,0))
    print("Prediction - Overall Mean Error:",np.mean(j2d_pred_ED))
    print("Projection - Overall Mean Error:",np.mean(j2d_proj_ED))
    print("Detection - Overall Mean Error:",np.mean(j2d_detect_ED))
    
    # draw 2d error bar and curves
    eval_errs = [j2d_pred_ED,j2d_proj_ED,j2d_detect_ED]
    eval_names = ['Predicted', 'Projected','Detected']
    metric_type = 'joint'#'max-frame','mean-frame','joint'
    fig = plt.figure(figsize=(16, 6))
    plt.figure(fig.number)
    #draw_error_bar(dataset, eval_errs, eval_names, fig)
    draw_2d_error_curve(eval_errs, eval_names, metric_type, fig)
    #plt.savefig(os.path.join(save_dir,'figures/{}_error.png'.format()))
    plt.savefig(os.path.join(save_dir,'error-pro_{0:.3f}-pre_{1:.3f}-detect_{2:.3f}.png'.format(np.mean(j2d_proj_ED),np.mean(j2d_pred_ED),np.mean(j2d_detect_ED))))
    print('save 2d error image')
    # save error to .txt
    
    savestr=os.path.join(save_dir, 'j2d_proj_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_proj_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    savestr=os.path.join(save_dir, 'j2d_pred_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_pred_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    savestr=os.path.join(save_dir, 'j2d_detect_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_detect_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    j2d_lists = [j2d_pred_list, j2d_proj_list, j2d_gt_list, j2d_detect_list, j2d_detect_con_list]
    j2d_lists_names = ['j2d_pred_list.txt','j2d_proj_list.txt','j2d_gt_list.txt','j2d_detect_list.txt','j2d_detect_con_list.txt']
    for ii in range(len(j2d_lists)):
        if len(j2d_lists[ii])>0:
            savestr=os.path.join(save_dir, j2d_lists_names[ii])
            with open(savestr,'w') as fp:
                for line in j2d_lists[ii]:
                    for l in line:
                        fp.write(str(l)+' ')
                    fp.write('\n')
    print("Write 2D Joints Error at:",save_dir)

def save_2d(examples, outputs, epoch, args):
    #save_dir = os.path.join(args.base_output_dir,'joint2d_result',str(epoch))
    #os.makedirs(save_dir, exist_ok=True)
    j2d_pred_ED, j2d_proj_ED, j2d_detect_ED = None, None, None
    if 'j2d_gt' in examples:
        if 'hm_j2d_list' in outputs:
            pred_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['hm_j2d_list'][-1])**2,2))#[8,21]
            j2d_pred_ED = pred_ED.cpu().detach().numpy().tolist()
        if 'j2d' in outputs:
            proj_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['j2d'])**2,2))#[8,21]
            j2d_proj_ED = proj_ED.cpu().detach().numpy().tolist()
        if 'open_2dj' in examples:
            detect_ED = torch.sqrt(torch.sum((examples['j2d_gt']-examples['open_2dj'])**2,2))#[8,21]
            j2d_detect_ED = detect_ED.cpu().detach().numpy().tolist()
    return j2d_pred_ED, j2d_proj_ED, j2d_detect_ED

def save_3d(examples, outputs):
    j3d_ED_list, j2d_ED_list = None, None
    if 'joints' in examples and 'joints' in outputs:
        j3d_ED = torch.sqrt(torch.sum((examples['joints']-outputs['joints'])**2,2))#[8,21]
        j3d_ED_list = j3d_ED.cpu().detach().numpy().tolist()
    if 'j2d_gt' in examples and 'j2d' in outputs:
        j2d_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['j2d'])**2,2))#[8,21]
        j2d_ED_list = j2d_ED.cpu().detach().numpy().tolist()
    return j3d_ED_list, j2d_ED_list

def log_3d_results(j3d_ED_list, j2d_ED_list, epoch, mode_train, logging):
    j3d_ED = np.asarray(j3d_ED_list)
    # j3d_per_joint = np.mean(j3d_ED,0)#[21]
    j3d_mean =np.mean(j3d_ED)#[1]
    j2d_ED = np.asarray(j2d_ED_list)
    # j2d_per_joint = np.mean(j2d_ED,0)#[21]
    j2d_mean =np.mean(j2d_ED)#[1]
    #logging.info("Epoch_{0}, Mean_j3d_error:{1}, Mean_j2d_error:{3}, Mean_per_j3d_error:{2}, Mean_per_j2d_error:{4}".format(epoch,j3d_mean,j3d_per_joint,j2d_mean,j2d_per_joint))
    trainOrEval = 'Training' if mode_train else 'Eval'
    logging.info("{3} Epoch_{0}, Mean_j3d_error (MPJPE):{1}, Mean_j2d_error:{2}".format(epoch, j3d_mean, j2d_mean, trainOrEval))


def visualize(mode_train,dat_name,epoch,idx_this,outputs,examples,args, op_outputs=None, writer=None, writer_tag='not-sure', console=None):
    # save images
    if mode_train:
        if idx_this % args.demo_freq == 0:
            with torch.no_grad():
                visualize_util.displadic(mode_train, args.obj_output, args.image_output, epoch, idx_this, examples, outputs, dat_name, op_outputs=op_outputs, writer=writer, writer_tag=writer_tag, console=console, img_wise_save=args.img_wise_save)
    else:
        if idx_this % args.demo_freq_evaluation == 0:
            with torch.no_grad():
                visualize_util.displadic(mode_train, args.obj_output, args.image_output, epoch, idx_this, examples, outputs, dat_name, op_outputs=op_outputs, writer=writer, writer_tag=writer_tag, console=console, img_wise_save=args.img_wise_save)
            if args.img_wise_save:
                visualize_util.multiview_render(args.image_output, outputs, epoch, idx_this)
                if op_outputs is not None:
                    op_outputs['faces'] = outputs['faces']
                    op_outputs['face_textures'] = outputs['face_textures']
                    op_outputs['render'] = outputs['render']
                    image_output = os.path.join(args.image_output, 'test-op')
                    os.makedirs(image_output, exist_ok=True)
                    visualize_util.multiview_render(image_output, outputs, epoch, idx_this)
    return 0
 

def write_to_tb(mode_train, writer,loss_dic, epoch, lr=None, is_val=False):
    if mode_train:
        writer.add_scalar('Learning_rate', lr, epoch)
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Train_'+loss_key, loss_dic[loss_key].sum().cpu().detach().numpy(), epoch)
    elif is_val:
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Val_'+loss_key, loss_dic[loss_key].sum().cpu().detach().numpy(), epoch)
    else:
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Test_'+loss_key, loss_dic[loss_key].sum().cpu().detach().numpy(), epoch)
    return 0


def mano_fitting(outputs,Ks=None, op_xyz_pred_list=[], op_verts_pred_list=[], dat_name='FreiHand',args=None):
    # 'pose', 'shape', 'scale','trans', 'rot'
    mano_shape = outputs['shape'].detach().clone()
    mano_pose = outputs['pose'].detach().clone()
    mano_trans = outputs['trans'].detach().clone()
    mano_scale = outputs['scale'].detach().clone()
    mano_rot = outputs['rot'].detach().clone()
    mano_shape.requires_grad = True
    mano_pose.requires_grad = True
    mano_trans.requires_grad = True
    mano_scale.requires_grad = True
    mano_rot.requires_grad = True
    mano_opt_params = [mano_shape, mano_pose,mano_trans,mano_scale,mano_rot]
    
    j2d_2dbranch = outputs['hm_j2d_list'][-1].detach().clone()#[b,21,2]
    j2d_2dbranch_con = torch.ones([j2d_2dbranch.shape[0],j2d_2dbranch.shape[1],1]).to(device)
    crit_l1 = nn.L1Loss()
    iter_total = 151
    batch_time = AverageMeter()
    end = time.time()
    for idx in range(iter_total):
        if idx < 51:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.01, betas=(0.9, 0.999))
        elif idx < 101:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.005, betas=(0.9, 0.999))
        else:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.0025, betas=(0.9, 0.999))
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(mano_rot, mano_pose, mano_shape)
        jv_ts = mano_trans.unsqueeze(1) + torch.abs(mano_scale.unsqueeze(2)) * jv[:,:,:]
        op_joints = jv_ts[:,0:21]
        op_verts = jv_ts[:,21:]
        
        if dat_name == 'FreiHand':
            op_joints = Mano2Frei(op_joints)
        loss = torch.zeros(1)
        # 2dj loss
        j2d = proj_func(op_joints, Ks)
        #reprojection_error = gmof(j2d - open_2dj, 100)
        # reprojection loss
        reprojection_distance = torch.sqrt(torch.sum((j2d_2dbranch-j2d)**2,2))
        #reprojection_distance = torch.where(reprojection_distance<5, reprojection_distance**2/10,reprojection_distance-2.5)
        reprojection_loss = args.lambda_j2d * torch.mean(reprojection_distance)
        # bone length loss
        op_bone_direc_loss = bone_direction_loss(j2d, j2d_2dbranch, j2d_2dbranch_con)
        op_bone_direc_loss = args.lambda_bone_direc * op_bone_direc_loss * 0.2

        # pose prior loss
        op_pose_loss = tsa_pose_loss(tsa_poses)
        op_pose_loss = args.lambda_pose * op_pose_loss * 3
        # shape prior loss
        op_shape_loss = torch_f.mse_loss(mano_shape, torch.zeros_like(mano_shape))
        op_shape_loss = args.lambda_shape * op_shape_loss
        # scale prior loss
        out_bone_length = torch.sqrt(torch.sum((op_joints[:,9, :] - op_joints[:,10, :])**2,1))
        op_scale_loss = crit_l1(out_bone_length,torch.ones_like(out_bone_length)*0.0282)
        op_scale_loss = args.lambda_mscale * op_scale_loss
        # triangle loss
        triangle_loss_fn = LaplacianLoss(torch.autograd.Variable(outputs['faces'][0]).cpu(),outputs['vertices'][0])
        triangle_loss = triangle_loss_fn(op_verts)
        triangle_loss = args.lambda_laplacian * triangle_loss
        #op_scale_loss = torch.zeros(1)
        total_loss = reprojection_loss + op_bone_direc_loss + op_pose_loss + op_shape_loss + op_scale_loss
        
        mano_optimizer.zero_grad()
        total_loss.backward()
        mano_optimizer.step()
        batch_time.update(time.time() - end)
        '''
        if idx%10==0:
            print('Iter: [{0}/{1}]\t' 'loss: {2:.4f}\t' 'Time {batch_time.val:.3f}\t'.format(idx,iter_total,total_loss.data.item(),batch_time=batch_time))
            #print("loss: {:.4f}".format(total_loss.data.item()))
            print("re2dj_loss: {0:.4f}; bone_dire_loss:{1:.4f}; pose_loss:{2:.8f}; shape_loss:{3:.8f}; scale_loss:{4:.8f}".format(reprojection_loss.data.item(),op_bone_direc_loss.data.item(), op_pose_loss.data.item(),op_shape_loss.data.item(),op_scale_loss.data.item()))
        '''
    op_outputs = {}
    op_outputs['j2d'] = j2d
    op_outputs['joints'] = op_joints
    op_outputs['vertices'] = op_verts
    if 'render' in outputs:
        op_re_img,op_re_depth,op_re_sil = outputs['render'](op_outputs['vertices'], outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
    else:
        op_re_img,op_re_depth,op_re_sil = None, None, None
    op_outputs['re_img'], op_outputs['re_deoth'], op_outputs['re_sil'] = op_re_img,op_re_depth,op_re_sil
    for i in range(op_outputs['joints'].shape[0]):
        if dat_name == "FreiHand":
            op_xyz_pred_list.append(op_outputs['joints'][i].cpu().detach().numpy())
        elif dat_name == "HO3D":
            output_joints_ho3d = Frei2HO3D(op_outputs['joints'])
            output_joints_ho3d = output_joints_ho3d.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            op_xyz_pred_list.append(output_joints_ho3d[i].cpu().detach().numpy())
        if 'vertices' in outputs:
            op_verts_pred_list.append(outputs['vertices'][i].cpu().detach().numpy())
    return op_outputs, op_xyz_pred_list, op_verts_pred_list