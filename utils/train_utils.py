import torch
import json
import os
import utils.visualize_util as visualize_util
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import logging
from rich import print


def load_model(model,optimizer,scheduler, args):
    current_epoch = 0
    
    #import pdb; pdb.set_trace()
    if args.pretrain_segmnet is not None:
        state_dict = torch.load(args.pretrain_segmnet)
        model.seghandnet.load_state_dict(state_dict['seghandnet'])
        #current_epoch = 0
        current_epoch = state_dict['epoch']
        print('loading the model from:', args.pretrain_segmnet)
        logging.info('pretrain_segmentation_model: %s' %args.pretrain_segmnet)
    if args.pretrain_model is not None:
        state_dict = torch.load(args.pretrain_model, map_location=args.device)
        #import pdb; pdb.set_trace()
        # dir(model)
        if 'encoder' in state_dict.keys() and hasattr(model,'encoder'):
            model.encoder.load_state_dict(state_dict['encoder'])
            print('load encoder')
        elif 'base_encoder' in state_dict.keys() and hasattr(model,'base_encoder'):
            model.base_encoder.load_state_dict(state_dict['base_encoder'])
            print('load base encoder')
        if 'decoder' in state_dict.keys() and hasattr(model,'hand_decoder'):
            model.hand_decoder.load_state_dict(state_dict['decoder'])
            print('load hand_decoder')
        elif 'hand_encoder' in state_dict.keys() and hasattr(model,'hand_encoder'):
            model.hand_encoder.load_state_dict(state_dict['hand_encoder'])
            print('load hand_encoder')
        if 'nimble_layer' in state_dict.keys() and hasattr(model,'nimble_layer'):
            model.nimble_layer.load_state_dict(state_dict['nimble_layer'])
            print('load nimble_layer')
        if 'heatmap_attention' in state_dict.keys() and hasattr(model,'heatmap_attention'):
            model.heatmap_attention.load_state_dict(state_dict['heatmap_attention'])
            print('load heatmap_attention')
        if 'rgb2hm' in state_dict.keys() and hasattr(model,'rgb2hm'):
            model.rgb2hm.load_state_dict(state_dict['rgb2hm'])
            print('load rgb2hm')
        if 'hm2hand' in state_dict.keys() and hasattr(model,'hm2hand'):
            model.hm2hand.load_state_dict(state_dict['hm2hand'])
        if 'mesh2pose' in state_dict.keys() and hasattr(model,'mesh2pose'):
            model.mesh2pose.load_state_dict(state_dict['mesh2pose'])
            print('load mesh2pose')
        
        if 'percep_encoder' in state_dict.keys() and hasattr(model,'percep_encoder'):
            model.percep_encoder.load_state_dict(state_dict['percep_encoder'])
        
        if 'texture_light_from_low' in state_dict.keys() and hasattr(model,'texture_light_from_low'):
            model.texture_light_from_low.load_state_dict(state_dict['texture_light_from_low'])
        if 'light_estimator' in state_dict.keys() and hasattr(model.module,'light_estimator'):
            model.light_estimator.load_state_dict(state_dict['light_estimator'])
        if 'textures' in args.train_requires and 'texture_estimator' in state_dict.keys():
            if hasattr(model,'renderer'):
                model.renderer.load_state_dict(state_dict['renderer'])
                print('load renderer')
            if hasattr(model,'texture_estimator'):
                model.texture_estimator.load_state_dict(state_dict['texture_estimator'])
                print('load texture_estimator')
            if hasattr(model,'pca_texture_estimator'):
                model.pca_texture_estimator.load_state_dict(state_dict['pca_texture_estimator'])
                print('load pca_texture_estimator')
        if 'lights' in args.train_requires and 'light_estimator' in state_dict.keys():
            if hasattr(model,'light_estimator'):
                model.light_estimator.load_state_dict(state_dict['light_estimator'])
                print('load light_estimator')
        print('loading the model from:', args.pretrain_model)
        logging.info('pretrain_model: %s' %args.pretrain_model)
        current_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])

        if hasattr(model,'texture_light_from_low') and args.pretrain_texture_model is not None:
            texture_state_dict = torch.load(args.pretrain_texture_model)
            model.texture_light_from_low.load_state_dict(texture_state_dict['texture_light_from_low'])
            print('loading the texture module from:', args.pretrain_texture_model)
    # load the pre-trained heat-map estimation model
    if hasattr(model,'rgb2hm') and args.pretrain_rgb2hm is not None:
        #util.load_net_model(args.pretrain_rgb2hm, model.rgb2hm.net_hm)
        #import pdb; pdb.set_trace()
        hm_state_dict = torch.load(args.pretrain_rgb2hm)
        model.rgb2hm.load_state_dict(hm_state_dict['rgb2hm'])
        print('load rgb2hm')
        print('loading the rgb2hm model from:', args.pretrain_rgb2hm)
    #import pdb; pdb.set_trace()
    return model, current_epoch, optimizer, scheduler


def save_model(model,optimizer,scheduler, epoch,current_epoch, args, console=None):
    state = {
        'args': args,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + current_epoch,
        'scheduler': scheduler.state_dict(),
        #'core': model.core.state_dict(),
    }
    save_file2 = None
    if args.save_mode == 'separately':
        postfix = epoch + current_epoch
    elif args.save_mode == 'only_latest':
        postfix = 'latest'
    if (epoch + current_epoch) % 20 == 0:
        save_file2 = os.path.join(args.state_output, f'texturehand_{epoch + current_epoch}.t7')

        
        
    if args.task == 'segm_train':
        state['seghandnet'] = model.module.seghandnet.state_dict()
        save_file = os.path.join(args.state_output, f'seghandnet_{postfix}.t7')
        print("[grey]Save model at:", save_file, "[/grey]")
        torch.save(state, save_file)
    elif args.task == 'train':
        if hasattr(model.module,'encoder'):
            state['encoder'] = model.module.encoder.state_dict()
        elif hasattr(model.module,'base_encoder'):
            state['base_encoder'] = model.module.base_encoder.state_dict()
        if hasattr(model.module,'hand_decoder'):
            state['decoder'] = model.module.hand_decoder.state_dict()
        elif hasattr(model.module,'hand_encoder'):
            state['hand_encoder'] = model.module.hand_encoder.state_dict()
        if hasattr(model.module,'nimble_layer'):
            state['nimble_layer'] = model.module.nimble_layer.state_dict()
        
        if hasattr(model.module,'heatmap_attention'):
            state['heatmap_attention'] = model.module.heatmap_attention.state_dict()
        if hasattr(model.module,'rgb2hm'):
            state['rgb2hm'] = model.module.rgb2hm.state_dict()
        if hasattr(model.module,'hm2hand'):
            state['hm2hand'] = model.module.hm2hand.state_dict()
        if hasattr(model.module,'mesh2pose'):
            state['mesh2pose'] = model.module.mesh2pose.state_dict()

        if hasattr(model.module,'percep_encoder'):
            state['percep_encoder'] = model.module.percep_encoder.state_dict()
        
        if hasattr(model.module,'texture_light_from_low'):
            state['texture_light_from_low'] = model.module.texture_light_from_low.state_dict()
        if hasattr(model.module,'light_estimator'):
            state['light_estimator'] = model.module.light_estimator.state_dict()

        if 'textures' in args.train_requires:
            if hasattr(model.module,'renderer'):
                state['renderer'] = model.module.renderer.state_dict()
            if hasattr(model.module,'texture_estimator'):
                state['texture_estimator'] = model.module.texture_estimator.state_dict()
            if hasattr(model.module,'pca_texture_estimator'):
                state['pca_texture_estimator'] = model.module.pca_texture_estimator.state_dict()
        if 'lights' in args.train_requires:
            if hasattr(model.module,'light_estimator'):
                state['light_estimator'] = model.module.light_estimator.state_dict()
                print("save light estimator")
        save_file = os.path.join(args.state_output, f'texturehand_{postfix}.t7')
        console.log(f"[bold green]Save model at {save_file}")
        torch.save(state, save_file)
        if save_file2 is not None:
            torch.save(state, save_file2)
            
    elif args.task == 'hm_train':
        state['rgb2hm'] = model.module.rgb2hm.state_dict()
        save_file = os.path.join(args.state_output, f'handhm_{postfix}.t7')
        print("Save model at:", save_file)
        torch.save(state, save_file)
    elif args.task == '2Dto3D':
        state['pose_lift_net'] = model.module.pose_lift_net.state_dict()
        save_file = os.path.join(args.state_output, f'pose_lift_net_{postfix}.t7')
        print("Save model at:", save_file)
        torch.save(state, save_file)

    return


def freeze_model_modules(model, args):
    if args.freeze_hm_estimator and hasattr(model.module,'rgb2hm'):
        visualize_util.rec_freeze(model.module.rgb2hm)
        print("Froze heatmap estimator")
    if args.only_train_regressor:
        if hasattr(model.module,'encoder'):
            visualize_util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module,'hand_decoder'):
            visualize_util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")
        if hasattr(model.module,'texture_estimator'):
            visualize_util.rec_freeze(model.module.texture_estimator)
            print("Froze texture estimator")
    if args.only_train_texture:
        if hasattr(model.module,'rgb2hm'):
            visualize_util.rec_freeze(model.module.rgb2hm)
            print("Froze rgb2hm")
        if hasattr(model.module,'encoder'):
            visualize_util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module,'hand_decoder'):
            visualize_util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")

def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('[grey]Dumped %d joints and %d verts predictions to %s[/grey]' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))













