import torch
import torch.nn as nn
import torch.nn.functional as torch_f

import utils.pytorch_ssim as pytorch_ssim
from utils.losses_util import bone_direction_loss, tsa_pose_loss, calc_laplacian_loss#image_l1_loss, iou_loss, ChamferLoss,

# !depreciated
def loss_func(examples, outputs, loss_used, dat_name, args) -> dict:
    loss_dic = {}
    device = examples['imgs'].device
    # heatmap integral loss: estimated 2d joints -> openpose 2d joints
    if 'hm_integral' in loss_used and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('hm_j2d_list' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        hm_integral_loss = torch.zeros(1).to(device)
        for hm_j2d in hm_j2d_list:
            hm_2dj_distance = torch.sqrt(torch.sum((examples['open_2dj']-hm_j2d)**2,2))#[b,21]
            open_2dj_con_hm = examples['open_2dj_con'].squeeze(2)
            hm_integral_loss += (torch.sum(hm_2dj_distance.mul(open_2dj_con_hm**2))/torch.sum((open_2dj_con_hm**2)))
        loss_dic['hm_integral'] = args.lambda_hm * hm_integral_loss
    else:
        loss_dic['hm_integral'] = torch.zeros(1)
    
    # (used in full supervision) estimated 2d joints -> gt 2d joints
    if 'hm_integral_gt' in loss_used and ('j2d_gt' in examples) and ('hm_j2d_list' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        hm_integral_loss = torch.zeros(1).to(device)
        for hm_j2d in hm_j2d_list:
            hm_2dj_distance0 = torch.sqrt(torch.sum((examples['j2d_gt']-hm_j2d)**2,2))#[b,21]
            open_2dj_con_hm0 = torch.ones_like(hm_2dj_distance0) # set confidence as 1
            hm_integral_loss += torch.sum(hm_2dj_distance0.mul(open_2dj_con_hm0**2))/torch.sum((open_2dj_con_hm0**2))
        loss_dic['hm_integral_gt'] = args.lambda_hm * hm_integral_loss
    else:
        loss_dic['hm_integral_gt'] = torch.zeros(1)
    
    # (used in full supervision) 2d joint loss: projected 2d joints -> gt 2d joints
    if 'j2d_gt' in examples and ('j2d' in outputs):
        joint_2d_loss = torch_f.mse_loss(examples['j2d_gt'], outputs['j2d'])
        joint_2d_loss = args.lambda_j2d_gt * joint_2d_loss
        loss_dic['joint_2d'] = joint_2d_loss
    else:
        loss_dic['joint_2d'] = torch.zeros(1)

    # open pose 2d joint loss: projected 2dj -> openpose 2dj
    if 'open_2dj' in loss_used and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('j2d' in outputs):
        open_2dj_distance = torch.sqrt(torch.sum((examples['open_2dj']-outputs['j2d'])**2,2))
        open_2dj_distance = torch.where(open_2dj_distance<5, open_2dj_distance**2/10,open_2dj_distance-2.5)
        keypoint_weights = torch.tensor([[2,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5]]).to(device).float()
        open_2dj_con0 = examples['open_2dj_con'].squeeze(2)
        open_2dj_con0 = open_2dj_con0.mul(keypoint_weights)
        open_2dj_loss = (torch.sum(open_2dj_distance.mul(open_2dj_con0**2))/torch.sum((open_2dj_con0**2)))
        open_2dj_loss = args.lambda_j2d * open_2dj_loss
        loss_dic['open_2dj'] = open_2dj_loss
    else:
        loss_dic['open_2dj'] = torch.zeros(1)
    # open pose 2d joint loss --- Downgrade Version
    if "open_2dj_de" in loss_used and ('open_2dj' in examples) and ('j2d' in outputs):
        open_2dj_loss = torch_f.mse_loss(examples['open_2dj'],outputs['j2d'])
        open_2dj_loss = args.lambda_j2d_de * open_2dj_loss
        loss_dic["open_2dj_de"] = open_2dj_loss
    else:
        loss_dic["open_2dj_de"] = torch.zeros(1)

    # (used in full supervision) 3D joint loss & Bone scale loss: 3dj -> gt 3dj
    
    if 'joints' in outputs and 'joints' in examples:
        joint_3d_loss = torch_f.mse_loss(outputs['joints'], examples['joints'])
        joint_3d_loss = args.lambda_j3d * joint_3d_loss
        loss_dic["joint_3d"] = joint_3d_loss
        # relative
        joint_3d_loss_norm = torch_f.mse_loss((outputs['joints']-outputs['joints'][:,9].unsqueeze(1)),(examples['joints']-examples['joints'][:,9].unsqueeze(1)))
        joint_3d_loss_norm = args.lambda_j3d_norm * joint_3d_loss_norm
        loss_dic["joint_3d_norm"] = joint_3d_loss_norm
    else:
        loss_dic["joint_3d"] = torch.zeros(1)
        loss_dic["joint_3d_norm"] = torch.zeros(1)

    # bone direction loss: 2d bones -> openpose bones
    if 'open_bone_direc' in loss_used and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('j2d' in outputs):
        open_bone_direc_loss = bone_direction_loss(outputs['j2d'], examples['open_2dj'], examples['open_2dj_con'])
        open_bone_direc_loss = args.lambda_bone_direc * open_bone_direc_loss
        loss_dic['open_bone_direc'] = open_bone_direc_loss
    else:
        loss_dic['open_bone_direc'] = torch.zeros(1)
    
    # (used in full supervision) projected 2d bones -> gt 2d bones
    if 'bone_direc' in loss_used and ('j2d_gt' in examples) and ('j2d' in outputs):
        j2d_con = torch.ones_like(examples['j2d_gt'][:,:,0]).unsqueeze(-1)
        bone_direc_loss = bone_direction_loss(outputs['j2d'], examples['j2d_gt'], j2d_con)
        bone_direc_loss = args.lambda_bone_direc * bone_direc_loss
        loss_dic['bone_direc'] = bone_direc_loss
    else:
        loss_dic['bone_direc'] = torch.zeros(1)
    
    # 2d-3d keypoints consistency loss: projected 2dj -> estimated 2dj
    if ('hm_j2d_list' in outputs) and ('j2d' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        kp_cons_distance = torch.sqrt(torch.sum((hm_j2d_list[-1]-outputs['j2d'])**2,2))
        kp_cons_distance = torch.where(kp_cons_distance<5, kp_cons_distance**2/10,kp_cons_distance-2.5)
        kp_cons_loss = torch.mean(kp_cons_distance)
        kp_cons_loss = args.lambda_kp_cons * kp_cons_loss
        loss_dic['kp_cons'] = kp_cons_loss
    else:
        loss_dic['kp_cons'] = torch.zeros(1)

    # mean scale regularization term
    if 'mscale' in loss_used and ('joints' in outputs):# and "joints" not in examples:
        out_bone_length = torch.sqrt(torch.sum((outputs['joints'][:,9, :] - outputs['joints'][:,10, :])**2,1))#check
        crit = nn.L1Loss()
        mscale_loss = crit(out_bone_length,torch.ones_like(out_bone_length)*0.0282)#check
        mscale_loss = args.lambda_mscale * mscale_loss
        loss_dic['mscale'] = mscale_loss
    else:
        loss_dic['mscale'] = torch.zeros(1)
    
    # (used in full supervision) GT scale loss
    if 'scale' in loss_used and ('joints' in outputs) and 'scales' in examples:
        if dat_name == 'FreiHand':
            cal_scale = torch.sqrt(torch.sum((outputs['joints'][:,9]-outputs['joints'][:,10])**2,1))
            scale_loss = torch_f.mse_loss(cal_scale, examples['scales'].to(device))
            scale_loss = args.lambda_scale * scale_loss
            loss_dic['scale'] = scale_loss
    else:
        loss_dic['scale'] = torch.zeros(1)

    # MANO pose regularization terms
    if 'tsa_poses' in outputs:
        pose_loss = tsa_pose_loss(outputs['tsa_poses'])
        pose_loss = args.lambda_pose * pose_loss
        loss_dic['tsa_poses'] = pose_loss
    else:
        loss_dic['tsa_poses'] = torch.zeros(1)
    
    # mesh texture regularization terms
    if 'mtex' in loss_used and ('textures' in outputs) and ('texture_con' in examples):
        textures = outputs['textures']
        std = torch.std(textures.view(textures.shape[0],-1,3),dim=1)#[b,3]
        mean = torch.mean(textures.view(textures.shape[0],-1,3),dim=1)
        textures_reg = (torch.where(textures>(mean.view(-1,1,1,1,1,3)+2*std.view(-1,1,1,1,1,3)),textures-mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))+torch.where(textures<(mean.view(-1,1,1,1,1,3)-2*std.view(-1,1,1,1,1,3)),-textures+mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))).squeeze()
        textures_reg = torch.sum(torch.mean(torch.mean(torch.mean(textures_reg,1),1),1).mul(examples['texture_con']*2))/torch.sum(examples['texture_con']**2)
        textures_reg = args.lambda_tex_reg * textures_reg
        loss_dic['mtex'] = textures_reg
    else:
        loss_dic['mtex'] = torch.zeros(1)

    # photometric loss
    if 're_img' in outputs and ('re_sil' in outputs) and ('texture_con' in examples):
        maskRGBs = outputs['maskRGBs']#examples['imgs'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
        re_img = outputs['re_img']
        crit = nn.L1Loss()

        # texture loss: rendered img -> masked original img
        #texture_loss = crit(re_img, maskRGBs).cpu()
        texture_con_this = examples['texture_con'].view(-1,1,1,1).repeat(1,re_img.shape[1],re_img.shape[2],re_img.shape[3])
        texture_loss = (torch.sum(torch.abs(re_img-maskRGBs).mul(texture_con_this**2))/torch.sum((texture_con_this**2)))
        texture_loss = args.lambda_texture * texture_loss
        loss_dic['texture'] = texture_loss

        # mean rgb loss
        #loss_mean_rgb = torch_f.mse_loss(torch.mean(maskRGBs),torch.mean(re_img)).cpu()
        loss_mean_rgb = (torch.sum(torch.abs(torch.mean(re_img.view(re_img.shape[0],-1),1)-torch.mean(maskRGBs.view(maskRGBs.shape[0],-1),1)).mul(examples['texture_con']**2))/torch.sum((examples['texture_con']**2)))
        loss_mean_rgb = args.lambda_mrgb * loss_mean_rgb
        loss_dic['mrgb'] = loss_mean_rgb

        # ssim texture loss
        ssim_tex = pytorch_ssim.ssim(re_img, maskRGBs)
        loss_ssim_tex = 1 - ssim_tex
        loss_ssim_tex = args.lambda_ssim_tex * loss_ssim_tex
        loss_dic['ssim_tex'] = loss_ssim_tex

        # ssim texture depth loss: ssim between rendered img -- rendered depth. ??? is it reasonable?
        ssim_tex_depth = pytorch_ssim.ssim(re_img, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_tex_depth = 1 - ssim_tex_depth
        loss_ssim_tex_depth = args.lambda_ssim_tex * loss_ssim_tex_depth
        loss_dic['ssim_tex_depth'] = loss_ssim_tex_depth
        
        # ssim depth loss: ssim between masked original img -- rendered depth. ???
        ssim_inrgb_depth = pytorch_ssim.ssim(maskRGBs, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_inrgb_depth = 1 - ssim_inrgb_depth
        loss_ssim_inrgb_depth = args.lambda_ssim_tex * loss_ssim_inrgb_depth
        loss_dic['ssim_inrgb_depth'] = loss_ssim_inrgb_depth
    else:
        loss_dic['texture'] = torch.zeros(1)
        loss_dic['mrgb'] = torch.zeros(1)
    
    # (fully supervision) silhouette loss: rendered sil -> gt sil
    if 're_sil' in outputs and 'segms_gt' in examples:
        crit = nn.L1Loss()
        sil_loss = crit(outputs['re_sil'], examples['segms_gt'].float())
        loss_dic['sil'] = args.lambda_silhouette * sil_loss
    else:
        loss_dic['sil'] = torch.zeros(1)

    # perceptual loss: rendered img -> gt img. not used at all.
    # if 'perc_features' in outputs and ('texture_con' in examples):
    #     perc_features = outputs['perc_features']
    #     batch_size = perc_features[0].shape[0]
    #     loss_percep_batch = torch.mean(torch.abs(perc_features[0]-perc_features[2]),1)+torch.mean(torch.abs(perc_features[1]-perc_features[3]).reshape(batch_size,-1),1)
    #     loss_percep = torch.sum(loss_percep_batch.mul( examples['texture_con']**2))/torch.sum(( examples['texture_con']**2))
    #     loss_percep = args.lambda_percep * loss_percep
    #     loss_dic['loss_percep'] = loss_percep
    # else:
    #     loss_dic['loss_percep'] = torch.zeros(1)
    
    # mesh laplacian regularization term
    if 'faces' in outputs and 'vertices' in outputs:
        # triangle_loss_fn = LaplacianLoss(torch.autograd.Variable(outputs['faces'][0]).cpu(),outputs['vertices'][0])
        # why [0]???
        # triangle_loss = triangle_loss_fn(outputs['vertices'])
        triangle_loss = calc_laplacian_loss(outputs['faces'], outputs['vertices'])
        triangle_loss = args.lambda_laplacian * triangle_loss
        loss_dic['triangle'] = triangle_loss
    else:
        loss_dic['triangle'] = torch.zeros(1)

    # mean shape loss: make shape towards 0???
    if 'shape' in outputs:
        shape_loss = torch_f.mse_loss(outputs['shape'], torch.zeros_like(outputs['shape']).to(device))
        shape_loss = args.lambda_shape * shape_loss
        loss_dic['mshape'] = shape_loss
    else:
        loss_dic['mshape'] = torch.zeros(1)
    
    return loss_dic

    
def loss_func_new(examples, outputs, loss_used, dat_name, args) -> dict:
    loss_dic = {}
    device = examples['imgs'].device

    if args.base_loss_fn == 'L1':
        base_loss_fn = nn.L1Loss()
    elif args.base_loss_fn == 'L2':
        base_loss_fn = torch_f.mse_loss
    
    # (used in full supervision) 2d joint loss: projected 2d joints -> gt 2d joints
    if 'joint_2d' in loss_used: 
        assert 'j2d_gt' in examples and ('j2d' in outputs), "Using joint_2d in losses, but j2d_gt or j2d are not provided."
        joint_2d_loss = base_loss_fn(examples['j2d_gt'], outputs['j2d'])
        joint_2d_loss = args.lambda_j2d_gt * joint_2d_loss
        loss_dic['joint_2d'] = joint_2d_loss

    # (used in full supervision) 3D joint loss & Bone scale loss: 3dj -> gt 3dj
    if 'joint_3d' in loss_used:
        assert 'joints' in outputs and 'joints' in examples, "Using joint_3d in losses, but joints or joints_gt are not provided."
        joint_3d_loss = base_loss_fn(outputs['joints'], examples['joints'])
        joint_3d_loss = args.lambda_j3d * joint_3d_loss
        loss_dic["joint_3d"] = joint_3d_loss
        # joint_3d_loss_norm = base_loss_fn((outputs['joints']-outputs['joints'][:,9].unsqueeze(1)),(examples['joints']-examples['joints'][:,9].unsqueeze(1)))
        # joint_3d_loss_norm = args.lambda_j3d_norm * joint_3d_loss_norm
        # loss_dic["joint_3d_norm"] = joint_3d_loss_norm

    # (used in full supervision) 3D verts loss: 3dj -> gt 3dj
    if 'vert_3d' in loss_used:
        assert 'mano_verts' in outputs and 'verts' in examples, "Using vert_3d in losses, but verts or verts_gt are not provided."
        vert_3d_loss = base_loss_fn(outputs['mano_verts'], examples['verts'])
        vert_3d_loss = args.lambda_vert_3d * vert_3d_loss
        loss_dic["vert_3d"] = vert_3d_loss

    # (used in full supervision) projected 2d bones -> gt 2d bones
    if 'bone_direc' in loss_used:
        assert ('j2d_gt' in examples) and ('j2d' in outputs), "Using bone_direc but j2d_gt not inputted or j2d not outputted"
        j2d_con = torch.ones_like(examples['j2d_gt'][:,:,0]).unsqueeze(-1)
        bone_direc_loss = bone_direction_loss(outputs['j2d'], examples['j2d_gt'], j2d_con)
        bone_direc_loss = args.lambda_bone_direc * bone_direc_loss
        loss_dic['bone_direc'] = bone_direc_loss
    

    # mean scale regularization term
    if 'mscale' in loss_used:
        assert ('joints' in outputs), "Using mscale but joints not outputted."
        out_bone_length = torch.sqrt(torch.sum((outputs['joints'][:,9, :] - outputs['joints'][:,10, :])**2,1))#check
        crit = nn.L1Loss()
        mscale_loss = crit(out_bone_length,torch.ones_like(out_bone_length)*0.0282)#check
        mscale_loss = args.lambda_mscale * mscale_loss
        loss_dic['mscale'] = mscale_loss
    
    # (used in full supervision) GT scale loss
    if 'scale' in loss_used:
        assert ('joints' in outputs) and 'scales' in examples, "Using scale as loss but joints not outputted or scales not inputted."
        if dat_name == 'FreiHand':
            cal_scale = torch.sqrt(torch.sum((outputs['joints'][:,9]-outputs['joints'][:,10])**2,1))
            scale_loss = torch_f.mse_loss(cal_scale, examples['scales'].to(device))
            scale_loss = args.lambda_scale * scale_loss
            loss_dic['scale'] = scale_loss

    # mesh texture regularization terms
    if 'mtex' in loss_used and ('textures' in outputs) and ('texture_con' in examples):
        textures = outputs['textures']
        std = torch.std(textures.view(textures.shape[0],-1,3),dim=1)#[b,3]
        mean = torch.mean(textures.view(textures.shape[0],-1,3),dim=1)
        textures_reg = (torch.where(textures>(mean.view(-1,1,1,1,1,3)+2*std.view(-1,1,1,1,1,3)),textures-mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))+torch.where(textures<(mean.view(-1,1,1,1,1,3)-2*std.view(-1,1,1,1,1,3)),-textures+mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))).squeeze()
        textures_reg = torch.sum(torch.mean(torch.mean(torch.mean(textures_reg,1),1),1).mul(examples['texture_con']*2))/torch.sum(examples['texture_con']**2)
        textures_reg = args.lambda_tex_reg * textures_reg
        loss_dic['mtex'] = textures_reg

    # photometric loss
    if 're_img' in outputs and ('re_sil' in outputs) and ('texture_con' in examples):
        maskRGBs = outputs['maskRGBs']#examples['imgs'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
        re_img = outputs['re_img']
        crit = nn.L1Loss()

        # texture loss: rendered img -> masked original img
        #texture_loss = crit(re_img, maskRGBs).cpu()
        texture_con_this = examples['texture_con'].view(-1,1,1,1).repeat(1,re_img.shape[1],re_img.shape[2],re_img.shape[3])
        texture_loss = (torch.sum(torch.abs(re_img-maskRGBs).mul(texture_con_this**2))/torch.sum((texture_con_this**2)))
        texture_loss = args.lambda_texture * texture_loss
        loss_dic['texture'] = texture_loss

        # mean rgb loss
        #loss_mean_rgb = torch_f.mse_loss(torch.mean(maskRGBs),torch.mean(re_img)).cpu()
        loss_mean_rgb = (torch.sum(torch.abs(torch.mean(re_img.view(re_img.shape[0],-1),1)-torch.mean(maskRGBs.view(maskRGBs.shape[0],-1),1)).mul(examples['texture_con']**2))/torch.sum((examples['texture_con']**2)))
        loss_mean_rgb = args.lambda_mrgb * loss_mean_rgb
        loss_dic['mrgb'] = loss_mean_rgb

        # ssim texture loss
        ssim_tex = pytorch_ssim.ssim(re_img, maskRGBs)
        loss_ssim_tex = 1 - ssim_tex
        loss_ssim_tex = args.lambda_ssim_tex * loss_ssim_tex
        loss_dic['ssim_tex'] = loss_ssim_tex

        # ssim texture depth loss: ssim between rendered img -- rendered depth. ??? is it reasonable?
        ssim_tex_depth = pytorch_ssim.ssim(re_img, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_tex_depth = 1 - ssim_tex_depth
        loss_ssim_tex_depth = args.lambda_ssim_tex * loss_ssim_tex_depth
        loss_dic['ssim_tex_depth'] = loss_ssim_tex_depth
        
        # ssim depth loss: ssim between masked original img -- rendered depth. ???
        ssim_inrgb_depth = pytorch_ssim.ssim(maskRGBs, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_inrgb_depth = 1 - ssim_inrgb_depth
        loss_ssim_inrgb_depth = args.lambda_ssim_tex * loss_ssim_inrgb_depth
        loss_dic['ssim_inrgb_depth'] = loss_ssim_inrgb_depth
    
    # (fully supervision) silhouette loss: rendered sil -> gt sil
    if 're_sil' in outputs and 'segms_gt' in examples:
        crit = nn.L1Loss()
        sil_loss = crit(outputs['re_sil'], examples['segms_gt'].float())
        loss_dic['sil'] = args.lambda_silhouette * sil_loss

    # perceptual loss: rendered img -> gt img. not used at all.
    # if 'perc_features' in outputs and ('texture_con' in examples):
    #     perc_features = outputs['perc_features']
    #     batch_size = perc_features[0].shape[0]
    #     loss_percep_batch = torch.mean(torch.abs(perc_features[0]-perc_features[2]),1)+torch.mean(torch.abs(perc_features[1]-perc_features[3]).reshape(batch_size,-1),1)
    #     loss_percep = torch.sum(loss_percep_batch.mul( examples['texture_con']**2))/torch.sum(( examples['texture_con']**2))
    #     loss_percep = args.lambda_percep * loss_percep
    #     loss_dic['loss_percep'] = loss_percep
    # else:
    #     loss_dic['loss_percep'] = torch.zeros(1)
    
    # mesh laplacian regularization term
    if 'triangle' in loss_used:
        assert 'faces' in outputs and 'verts' in outputs, "Using triangle as loss but faces or verts are not outputted."
        # triangle_loss_fn = LaplacianLoss(torch.autograd.Variable(outputs['faces'][0]).cpu(),outputs['vertices'][0])
        # why [0]???
        # triangle_loss = triangle_loss_fn(outputs['vertices'])
        triangle_loss = calc_laplacian_loss(outputs['faces'], outputs['verts'])
        triangle_loss = args.lambda_laplacian * triangle_loss
        loss_dic['triangle'] = triangle_loss

    # min shape loss: make shape towards 0
    if 'mshape' in loss_used:
        assert 'shape_params' in outputs, "Using mshape as loss but shape_params not outputted."
        shape_loss = torch_f.mse_loss(outputs['shape_params'], torch.zeros_like(outputs['shape_params']).to(device))
        shape_loss = args.lambda_shape * shape_loss
        loss_dic['mshape'] = shape_loss

    # min pose loss: make shape towards 0
    if 'mpose' in loss_used:
        assert 'pose_params' in outputs, "Using mpose as loss but pose_params not outputted."
        pose_loss = torch_f.mse_loss(outputs['pose_params'], torch.zeros_like(outputs['pose_params']).to(device))
        # pose_loss = outputs['pose_params'].pow(2).sum(dim=-1).sqrt().mean()*10  
        pose_loss = args.lambda_pose * pose_loss
        loss_dic['mpose'] = pose_loss
    
    return loss_dic