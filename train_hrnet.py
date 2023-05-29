import logging
import os
from rich import print
from rich.console import Console

import numpy as np
import models as models
import models_res_nimble as models_new
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as torch_f
import lpips
import utils.pytorch_ssim as pytorch_ssim

from torch.utils.tensorboard import SummaryWriter

from options import train_options
from losses import LossFunction
from data.dataset import get_dataset

from utils.train_utils import *
from utils.concat_dataloader import ConcatDataloader
from utils.traineval_util import data_dic, log_3d_results, save_2d_result,save_2d, mano_fitting, save_3d, trans_proj_j2d, visualize, write_to_tb, Mano2Frei, ortho_project
from utils.fh_utils import AverageMeter,EvalUtil, Frei2HO3D


console = Console()
test_log = {}

def train_an_epoch(mode_train, dat_name, epoch, train_loader, model, optimizer, requires, args, writer=None):
    if mode_train:
        model.train()
        set_name = 'training'
    else:
        model.eval()
        set_name = 'evaluation'

    batch_time = AverageMeter()
    end = time.time()
    # Init output containers
    
    evalutil = EvalUtil()
    xyz_pred_list, verts_pred_list = list(), list()
    # op_xyz_pred_list, op_verts_pred_list = list(), list()
    j2d_pred_ED_list,  j2d_proj_ED_list, j2d_detect_ED_list = list(), list(), list() 
    texture_metric_list = list()

    
    for idx, (sample) in enumerate(train_loader):
        # Get batch data
        examples = data_dic(sample, dat_name, set_name, args)
        del sample
        
        if set_name == 'evaluation' and dat_name == 'HO3D':
            root_xyz = examples['root_xyz'].unsqueeze(1)
        else:
            root_xyz = examples['joints'][:, args.ROOT, :].unsqueeze(1)
        
        # root_xyz = examples['joints'][:, args.ROOT, :].unsqueeze(1)
        # Use the network to predict the outputs
        outputs = model(examples['imgs'], Ks=examples['Ps'], root_xyz=root_xyz)

        # ** positions are relative to middle root.
        if set_name != 'evaluation' and dat_name != 'HO3D':
            examples['joints'] = examples['joints'] - root_xyz
            if 'verts' in examples:
                examples['verts'] = examples['verts'] - root_xyz

        if dat_name == 'Dart':
            # Projection transformation, project joints to 2D
            if 'joints' in outputs:
                j2d = ortho_project(outputs['joints'].float(), examples['ortho_intr'].float())
                j2d = torch.FloatTensor(j2d).to(args.device)
                outputs.update({'j2d': j2d})
                if args.hand_model == 'nimble':
                    nimble_j2d = ortho_project(outputs['nimble_joints'].float(), examples['ortho_intr'].float())
                    nimble_j2d = torch.FloatTensor(nimble_j2d).to(args.device)
                    outputs.update({'nimble_j2d': nimble_j2d})
        else:
            # Projection transformation, project joints to 2D
            if 'joints' in outputs:
                j2d = trans_proj_j2d(outputs, examples['Ks'], root_xyz=root_xyz) # do not need scale
                outputs.update({'j2d': j2d})
                if args.hand_model == 'nimble':
                    # nimble_j2d = trans_proj_j2d(outputs, examples['Ks'], examples['scales'], root_xyz=root_xyz, which_joints='nimble_joints')
                    nimble_j2d = trans_proj_j2d(outputs, examples['Ks'], root_xyz=root_xyz, which_joints='nimble_joints')
                    outputs.update({'nimble_j2d': nimble_j2d})
        
        # ===================================
        #      Compute and backward loss
        # ===================================
        loss_used = args.losses
        loss = torch.zeros(1).float().to(args.device)

        if mode_train: # only compute loss for training
            loss_dic = loss_func(examples, outputs, loss_used, dat_name, args)
            for loss_key in loss_used:
                # if loss_dic[loss_key]>0 and (not torch.isnan(loss_dic[loss_key]).sum()):
                loss += loss_dic[loss_key]
                    #print(loss_key,loss_dic[loss_key],loss_dic[loss_key].device)
        else:
            loss_dic = {}
            
        loss_dic['loss']=loss
        if loss < 1e-10 and len(loss_dic.keys())>1:
            print('loss is less than 1e-10')
            continue
        
        if mode_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ================================
        #       print and save results
        # ================================
        # save 3D pred joints
        if args.save_3d or not mode_train: # only save the pred results for evaluation
            # xyz_preds = outputs['joints'].cpu().detach().numpy()
            # xyz_preds = np.split(xyz_preds, xyz_preds.shape[0])
            # for i in xyz_preds:
            #     xyz_pred_list.append(i.squeeze())
            # vert_preds = outputs['mano_verts'].cpu().detach().numpy()
            # vert_preds = np.split(vert_preds, vert_preds.shape[0])
            # for i in vert_preds:
            #     verts_pred_list.append(i.squeeze())
            # j3d_ED_list, j2d_ED_list = save_3d(examples, outputs) # Euclidean distances between each joint-pair
            # log_3d_results(j3d_ED_list, j2d_ED_list, epoch, mode_train, logging)
            # del j3d_ED_list, j2d_ED_list 
            for i in range(outputs['joints'].shape[0]):
                #import pdb; pdb.set_trace()
                if dat_name == "FreiHand":
                    xyz_pred_list.append(outputs['joints'][i].cpu().detach().numpy())
                elif dat_name == "HO3D":
                    output_joints_ho3d = Frei2HO3D(outputs['joints'])
                    #import pdb; pdb.set_trace()
                    output_joints_ho3d = output_joints_ho3d.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
                    xyz_pred_list.append(output_joints_ho3d[i].cpu().detach().numpy()) 
        # save 2D results
        if args.save_2d:
            # square errors?
            j2d_pred_ED, j2d_proj_ED, j2d_detect_ED = save_2d(examples, outputs, epoch, args)
            j2d_pred_ED_list.append(j2d_pred_ED)
            j2d_proj_ED_list.append(j2d_proj_ED)
            j2d_detect_ED_list.append(j2d_detect_ED)

        # compute texture metric
        if not mode_train and args.render:
            if dat_name == 'HO3D':
                maskRGBs = examples['imgs'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
                mask_re_img = outputs['re_img'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
            else:
                maskRGBs = examples['segms_gt'].unsqueeze(1) * examples['imgs'] #examples['imgs'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
                mask_re_img = outputs['re_img'] * examples['segms_gt'].unsqueeze(1) # (outputs['re_sil']/255.0).repeat(1,3,1,1)
            psnr = -10 * loss_func.MSE_loss(mask_re_img, maskRGBs).log10().item()
            ssim = pytorch_ssim.ssim(mask_re_img, maskRGBs).item()
            lpips = lpips_loss(mask_re_img * 2 - 1, maskRGBs * 2 - 1).mean().item()
            l1 = loss_func.L1_loss(mask_re_img, maskRGBs).mean().item()
            l2 = loss_func.MSE_loss(mask_re_img, maskRGBs).mean().item()
            texture_metric_list.append({'psnr':psnr, 'ssim':ssim, 'lpips':lpips, 'l1': l1, 'l2': l2})


        # Save visualization and print information
        batch_time.update(time.time() - end)
        
        visualize(mode_train, dat_name, epoch, idx, outputs, examples, args, writer=writer, writer_tag=set_name, console=console)
        # Print information
        if idx % args.print_freq == 0:
            if optimizer is not None:
                lr_current = optimizer.param_groups[0]['lr']
            else:
                lr_current = 0
            if not mode_train:
                prefix_test = '[bold yellow]Test [/bold yellow]'
            else:
                prefix_test = ''
            console.log('{prefix_test}Epoch: [{0}/{tot_epoch}]\t'
                'Iter: [{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '[bold red]Loss {loss:.5f}[/bold red]\t'
                'dataset: {dataset:6}\t'
                'lr {lr:.7f}\t'.format(epoch, idx, len(train_loader),
                                        batch_time=batch_time, loss=loss.data.item(), dataset=dat_name,
                                        lr=lr_current, prefix_test=prefix_test, tot_epoch=args.total_epochs))
            console.log(f"Loss backward:\t{', '.join(['{0}: {1:6f}'.format(loss_item,loss_data.sum()) for loss_item,loss_data in loss_dic.items() if (loss_item in loss_used)])}")

            #print("Loss all:\t",['{0}:{1:6f};'.format(loss_item, loss_dic[loss_item].sum().data.item()) for loss_item in loss_dic])
            #print("j3d loss:{0:.4f}; j2d loss:{1:.4f};shape loss:{2:.6f}; pose loss:{3:.6f}; render loss:{4:.6f}; sil loss:{5:.6f}; depth loss:{6:.5f}; render ssim loss:{7:.5f}; depth ssim loss:{8:.5f}; open j2d loss:{9:.5f}; mesh tex std:{10:.10f}; scale loss:{11:.5f}; bone direct loss:{12:.5f}; laplacian loss:{13:.6f}; hm loss:{14:.6f}; kp consistency loss:{15:.6f}; percep loss:{16:.6f}".format(joint_3d_loss.data.item(),joint_2d_loss.data.item(), shape_loss.data.item(),pose_loss.data.item(),texture_loss.data.item(), silhouette_loss.data.item(), depth_loss.data.item(), loss_ssim_tex.data.item(), loss_ssim_depth.data.item(), open_2dj_loss.data.item(), textures_reg.data.item(), mscale_loss.data.item(), open_bone_direc_loss.data.item(),triangle_loss.data.item(),hm_loss.data.item(),kp_cons_loss.data.item(),loss_percep.data.item()))
        # write to tensorboard
        if writer is not None:
            with torch.no_grad():
                write_to_tb(mode_train, writer, loss_dic, epoch, lr=optimizer.param_groups[0]['lr'])

    # after one epoch....
    # dump results
    if dat_name == 'FreiHand':
        if mode_train:
            pred_out_path = os.path.join(args.pred_output,'train',str(epoch))
            if args.save_3d:
                os.makedirs(pred_out_path, exist_ok=True)
                pred_out_path_0 = os.path.join(pred_out_path,'pred.json')
                dump(pred_out_path_0, xyz_pred_list, verts_pred_list)
        else: # for evaluation
            # ================================
            #          Evaluation
            # ================================
            pred_out_path = os.path.join(args.pred_output,'test',str(epoch))
            if epoch%args.save_interval==0 and epoch>0:
                os.makedirs(pred_out_path, exist_ok=True)
                pred_out_path_0 = os.path.join(pred_out_path,'pred.json')
                # dump(pred_out_path_0, xyz_pred_list, verts_pred_list)
                # pred_out_op_path = os.path.join(pred_out_path,'pred_op.json')
                # dump(pred_out_op_path, op_xyz_pred_list, op_verts_pred_list)

                # ---- evaluation: MPJPE and MPVPE after alignment --------
                # load eval annotations
                gt_path = args.freihand_base_path
                xyz_list, verts_list = json_load(os.path.join(gt_path, 'evaluation_xyz.json')), json_load(os.path.join(gt_path, 'evaluation_verts.json'))
                pose_align_all = []
                vert_align_all = []
                pose_3d = np.array(xyz_pred_list)
                vert_3d = np.array(verts_pred_list)
                pose_3d_gt = np.array(xyz_list)
                vert_3d_gt = np.array(verts_list)

                for idx in range(pose_3d.shape[0]):
                    #align prediction
                    pose_pred_aligned=align_w_scale(pose_3d_gt[idx], pose_3d[idx])
                    vert_pred_aligned=align_w_scale(vert_3d_gt[idx], vert_3d[idx])
                    pose_align_all.append(pose_pred_aligned)
                    vert_align_all.append(vert_pred_aligned)
                pose_align_all = torch.from_numpy(np.array(pose_align_all)).cuda()
                vert_align_all = torch.from_numpy(np.array(vert_align_all)).cuda()
                pose_3d_gt = torch.from_numpy(pose_3d_gt).cuda()
                vert_3d_gt = torch.from_numpy(vert_3d_gt).cuda()

                pose_3d_loss = torch.linalg.norm((pose_align_all - pose_3d_gt), ord=2,dim=-1)
                vert_3d_loss = torch.linalg.norm((vert_align_all - vert_3d_gt), ord=2,dim=-1)
                pose_3d_loss = (np.concatenate(pose_3d_loss.detach().cpu().numpy(),axis=0)).mean()
                vert_3d_loss = (np.concatenate(vert_3d_loss.detach().cpu().numpy(),axis=0)).mean()

                console.log(f"Evaluation pose 3d: {pose_3d_loss * 100.0:.6f} cm, vert 3d: {vert_3d_loss * 100.0:.6f} cm")
                test_log[epoch] = [pose_3d_loss.item(), vert_3d_loss.item()]

                best_MPJPE = min(test_log.values(), key=lambda x: x[0])[0]
                best_results = [k for k, v in test_log.items() if v[0] == best_MPJPE]
                best_epoch = best_results[0]
                best_MPJPE, best_MPVPE = test_log[best_epoch]
                console.log(f'[bold green]Best MPJPE: {best_MPJPE * 100:.6f} cm, MPVPE: {best_MPVPE * 100:.6f}, Epoch: {best_epoch}\n')

                if writer is not None:
                    with torch.no_grad():
                        writer.add_scalar('eval/pose_3d_loss', pose_3d_loss.item(), epoch)
                        writer.add_scalar('eval/vert_3d_loss', vert_3d_loss.item(), epoch)

                # ----- evaluation: texture metrics --------        
                if args.render:
                    psnr = np.mean([r['psnr'] for r in texture_metric_list])
                    ssim = np.mean([r['ssim'] for r in texture_metric_list])
                    lpips = np.mean([r['lpips'] for r in texture_metric_list])
                    l1 = np.mean([r['l1'] for r in texture_metric_list])
                    l2 = np.mean([r['l2'] for r in texture_metric_list])
                    console.log(f'[bold green]PSNR:  {psnr:8.4f}, SSIM:  {ssim:8.4f}, LPIPS: {lpips:8.4f}, l1: {l1:8.4f}, l2: {l2:8.4f}\n')

                    if writer is not None:
                        with torch.no_grad():
                            writer.add_scalar('eval/psnr', psnr, epoch)
                            writer.add_scalar('eval/ssim', ssim, epoch)
                            writer.add_scalar('eval/lpips', lpips, epoch)
                            writer.add_scalar('eval/l1', l1, epoch)
                            writer.add_scalar('eval/l2', l2, epoch)

        if args.save_2d:
            save_2d_result(j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list, args=args, epoch=epoch)
    
    if dat_name == 'HO3D':
        if mode_train:
            pred_out_path = os.path.join(args.pred_output,'train',str(epoch))
            if args.save_3d:
                os.makedirs(pred_out_path, exist_ok=True)
                pred_out_path_0 = os.path.join(pred_out_path,'pred.json')
                dump(pred_out_path_0, xyz_pred_list, verts_pred_list)
        else: # for evaluation
            # ================================
            #          Evaluation
            # ================================
            pred_out_path = os.path.join(args.pred_output,'test',str(epoch))
            # if epoch%args.save_interval==0 and epoch>0:
            os.makedirs(pred_out_path, exist_ok=True)
            pred_out_path_0 = os.path.join(pred_out_path,'pred.json')
            # HO3D dump evaluation result for online evaluation
            dump(pred_out_path_0, xyz_pred_list, verts_pred_list)
            # pred_out_op_path = os.path.join(pred_out_path,'pred_op.json')
            # dump(pred_out_op_path, op_xyz_pred_list, op_verts_pred_list)
            # ----- evaluation: texture metrics --------        
            if args.render:
                psnr = np.mean([r['psnr'] for r in texture_metric_list])
                ssim = np.mean([r['ssim'] for r in texture_metric_list])
                lpips = np.mean([r['lpips'] for r in texture_metric_list])
                l1 = np.mean([r['l1'] for r in texture_metric_list])
                l2 = np.mean([r['l2'] for r in texture_metric_list])
                console.log(f'[bold green]PSNR:  {psnr:8.4f}, SSIM:  {ssim:8.4f}, LPIPS: {lpips:8.4f}, l1: {l1:8.4f}, l2: {l2:8.4f}\n')

                if writer is not None:
                    with torch.no_grad():
                        writer.add_scalar('eval/psnr', psnr, epoch)
                        writer.add_scalar('eval/ssim', ssim, epoch)
                        writer.add_scalar('eval/lpips', lpips, epoch)
                        writer.add_scalar('eval/l1', l1, epoch)
                        writer.add_scalar('eval/l2', l2, epoch)


def train(base_path, set_name=None, writer = None, optimizer = None, scheduler = None):
    """
        Main loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """

    # ==============================
    #        prepare dataset
    # ==============================
    with console.status("Preparing dataset...", spinner="bounce"):
        assert set_name is not None, "Mode is not provided. Should be training or evaluation."
        if args.controlled_exp:
            # Use subset of datasets so that final dataset size is constant
            limit_size = int(args.controlled_size / len(args.train_datasets))
        else:
            limit_size = None

        if 'training' in set_name:
            # initialize train datasets
            train_loaders = []
            for dat_name in args.train_datasets:# iteration = min(dataset_len)/batch_size; go each dataset at a batchsize
                if dat_name == 'FreiHand':
                    if len(args.train_queries_frei)>0:
                        train_queries = args.train_queries_frei
                    else:
                        train_queries = args.train_queries
                    base_path = args.freihand_base_path
                elif dat_name == 'RHD':
                    if len(args.train_queries_rhd)>0:
                        train_queries = args.train_queries_rhd
                    else:
                        train_queries = args.train_queries
                    base_path = args.rhd_base_path
                elif (dat_name == 'Obman') or (dat_name == 'Obman_hand'):
                    train_queries = args.train_queries
                elif dat_name == 'HO3D':
                    if len(args.train_queries_ho3d)>0:
                        train_queries = args.train_queries_ho3d
                    else:
                        train_queries = args.train_queries
                    base_path = args.ho3d_base_path
                elif dat_name == 'Dart':
                    if len(args.train_queries_dart)>0:
                        train_queries = args.train_queries_dart
                    else:
                        train_queries = args.train_queries
                    base_path = args.dart_base_path
                
                train_dat = get_dataset(
                    dat_name,
                    'training',#set_name,
                    base_path,
                    queries = train_queries,
                    train = True,
                    limit_size=limit_size,
                    if_use_j2d = args.four_channel
                    #transform=transforms.Compose([transforms.Rescale(256),transforms.ToTensor()]))
                )
                print("Training dataset size: {}".format(len(train_dat)))
                # Initialize train dataloader
                # This is only for generating pred.json and for evaluation the training metrics
                if args.save_3d:
                    train_loader0 = torch.utils.data.DataLoader(
                        train_dat,
                        batch_size=args.train_batch,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=False,
                    )
                else:
                    train_loader0 = torch.utils.data.DataLoader(
                        train_dat,
                        batch_size=args.train_batch,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                    )
                train_loaders.append(train_loader0)
            train_loader = ConcatDataloader(train_loaders)
        #if 'evaluation' in set_name:
        val_loaders = []
        for dat_name_val in args.val_datasets:
            if dat_name_val == 'FreiHand':
                val_queries = args.val_queries
                base_path = args.freihand_base_path
            elif dat_name_val == 'RHD':
                val_queries = args.val_queries
                base_path = args.rhd_base_path
            elif dat_name_val == 'HO3D':
                val_queries = args.val_queries
                base_path = args.ho3d_base_path
            elif dat_name_val == 'Dart':
                val_queries = args.val_queries
                base_path = args.dart_base_path
            val_dat = get_dataset(
                dat_name_val,
                'evaluation',
                base_path,
                queries = val_queries,
                train = False,
                limit_size=limit_size,
                #transform=transforms.Compose([transforms.Rescale(256),transforms.ToTensor()]))
            )
            print("Validation dataset size: {}".format(len(val_dat)))
            val_loader = torch.utils.data.DataLoader(
                val_dat,
                batch_size=args.val_batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_loaders.append(val_loader)
        val_loader = ConcatDataloader(val_loaders)

        #current_epoch = 0
        if len(args.train_datasets) == 1:
            dat_name = args.train_datasets[0]#dat_name
        else:
            dat_name = args.train_datasets
        
        # for saving visualization outputs
        if 'training' in set_name:
            args.obj_output = os.path.join(args.obj_output,'train')
            args.image_output = os.path.join(args.image_output, 'train')
        else:
            args.obj_output = os.path.join(args.obj_output,'test')
            args.image_output = os.path.join(args.image_output, 'test')
        os.makedirs(args.obj_output, exist_ok=True)
        os.makedirs(args.image_output, exist_ok=True)

    # =======================================
    #         Training loop
    # =======================================
    if 'training' in set_name:

        with console.status("Training...", spinner="monkey") as status:
            for epoch in range(1, args.total_epochs + 1 - current_epoch):
                # step the lambda...
                for i, lambda_pose_step in enumerate(args.lambda_pose_steps):
                    if lambda_pose_step <= epoch + current_epoch:
                        args.lambda_pose = args.lambda_pose_list[i + 1]
                for i, lambda_j2d_gt_step in enumerate(args.lambda_j2d_gt_steps):
                    if lambda_j2d_gt_step <= epoch + current_epoch:
                        args.lambda_j2d_gt = args.lambda_j2d_gt_list[i + 1]
                for i, lambda_shape_step in enumerate(args.lambda_shape_steps):
                    if lambda_shape_step <= epoch + current_epoch:
                        args.lambda_shape = args.lambda_shape_list[i + 1]
                for i, lambda_tex_reg_step in enumerate(args.lambda_tex_reg_steps):
                    if lambda_tex_reg_step <= epoch + current_epoch:
                        args.lambda_tex_reg = args.lambda_tex_reg_list[i + 1]

                status.update(status="Training...", spinner="monkey")
                mode_train = True
                requires = args.train_requires
                args.train_batch = args.train_batch
                train_an_epoch(mode_train, dat_name, epoch + current_epoch, train_loader, model, optimizer, requires, args, writer)
                torch.cuda.empty_cache()

                status.update(status="[bold yellow] Testing...", spinner="weather")
                if (epoch + current_epoch) % args.save_interval == 0:
                # save model and test
                    if args.if_test:
                        # test
                        mode_train = False
                        requires = args.test_requires
                        args.train_batch = args.val_batch
                        train_an_epoch(mode_train, dat_name_val, epoch + current_epoch, val_loader, model, optimizer, requires, args, writer)
                        torch.cuda.empty_cache()
                    save_model(model,optimizer,scheduler, epoch,current_epoch, args, console=console)   
                scheduler.step()

    elif 'evaluation' in set_name:
        mode_train = False
        requires = args.test_requires
        optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), weight_decay=0)#
        #epoch = 0
        #current_epoch = 0
        #save_model(model,optimizer,epoch,current_epoch, args)
        train_an_epoch(mode_train, dat_name_val, current_epoch, val_loader, model, None, requires, args, writer)
        print("Finish write prediction. Good luck!")

    print("Done!")
    
if __name__ == '__main__':
    # ==================================
    #          prepare arguments
    # ==================================
    args = train_options.parse()
    
    if args.config_json is not None:
        print(f'Loading arguments from config_json file: {args.config_json}')
        with open(args.config_json, "r") as f:
            json_dic = json.load(f)
            for parse_key, parse_value in json_dic.items():
                setattr(args, parse_key, parse_value)
    
    args = train_options.make_output_dir(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.ROOT = 9
    args.ROOT_NIMBLE = 11
    args.lambda_pose = args.lambda_pose_list[0]
    args.lambda_shape = args.lambda_shape_list[0]
    args.lambda_j2d_gt = args.lambda_j2d_gt_list[0]
    args.lambda_tex_reg = args.lambda_tex_reg_list[0]

    if args.is_write_tb:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               args.writer_topic+datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir= log_dir)
        print(datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        writer = None
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(args.base_output_dir, 'train.log'), level=logging.INFO)
    logging.info("=====================================================")


    # ==================================
    #          initialize model
    # ==================================

    if args.new_model:
        print("Using new model... Equipping Resnet and NIMBLE!!")
        model = models_new.Model(ifRender=args.render, device=args.device, if_4c=args.four_channel, hand_model=args.hand_model, use_mean_shape=args.use_mean_shape, pretrain=args.pretrain,
                                 root_id=args.ROOT, root_id_nimble=args.ROOT_NIMBLE,
                                 ifLight=args.light_estimation)
    else:
        model = models.Model(args=args)
    model.to(args.device)
    
    if 'training' in args.mode:
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), weight_decay=0)
        elif args.optimizer == "AdamW":
            optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif 'evaluation' in args.mode:
        optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    
    model, current_epoch, optimizer, scheduler = load_model(model, optimizer, scheduler, args)
    if args.force_init_lr > 0: # default is -1, means not using this
        optimizer.param_groups[0]['lr'] = args.force_init_lr

    model = nn.DataParallel(model.cuda())

    loss_func = LossFunction()
    lpips_loss = lpips.LPIPS(net="alex").to(args.device)

    # Optionally freeze parts of the network
    freeze_model_modules(model, args)

    # call with a predictor function
    train(
        args.base_path,
        set_name=args.mode,
        writer = writer,
        optimizer = optimizer,
        scheduler = scheduler
    )
    if writer is not None:
        writer.close()