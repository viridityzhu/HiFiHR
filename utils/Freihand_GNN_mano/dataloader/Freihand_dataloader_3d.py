import numpy as np
import cv2
# from pycocotools.coco import COCO
from torch.utils.data import Dataset
import json
import os
import time
from PIL import Image
import random
import torchvision
from torchvision import transforms
import copy
import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz


def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'

class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)

def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return cv2.imread(img_rgb_path)


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)
    scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
    verts_path = os.path.join(base_path, '%s_verts.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)
    scale_list = json_load(scale_path)
    verts_list = json_load(verts_path)

    # should have all the same length
    # assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'
    assert len(K_list) == len(scale_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    # return list(zip(K_list, mano_list, xyz_list, scale_list))
    return list(zip(K_list, xyz_list, scale_list, verts_list,mano_list))

def rotate(origin, point, angle, ):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox +   math.cos(angle) * (px - ox) -   math.sin(angle) * (py - oy)
    qy = oy +   math.sin(angle) * (px - ox) +    math.cos(angle) * (py - oy)

    return qx, qy        


def processing_augmentation(image,pose3d,verts):  #scale, transiltion， uv argumentation is different from the xyz vertices
    randScaleImage = np.random.uniform(low=0.8, high=1.0)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0] #change the rotation to
    rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
                                     randScaleImage)  # change image later together with translation

    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    #rotation
    (pose3d[:, 0], pose3d[:, 1]) = rotate((pose3d[9,0],pose3d[9,1]), (pose3d[:, 0], pose3d[:, 1]), randAngle)
    (verts[:, 0], verts[:, 1]) = rotate((pose3d[9,0],pose3d[9,1]), (verts[:, 0], verts[:, 1]), randAngle)
   

    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    image = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    image = np.reshape(image, [256, 256, 3])
    return image, pose3d,verts

def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]],
                                     [0, 1, L - mid[1]]])

    img_list_out = []
    for img in img_list:
        img_list_out.append(cv2.warpAffine(img, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, label2d_list_out, camera

def rgb_processing(rgb_img):
    # in the rgb image we add pixel noise in a channel-wise manner
    noise_factor = 0.4
    pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    return rgb_img
class FreiHAND(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode in ['training', 'evaluation'], 'mode error'
        # load annotations
        self.anno_all = load_db_annotation(root, 'training')
        self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.anno_all)*4
        #return 100       
    def __getitem__(self, id):
        idx = id % 32560
        img_idx = id // 32560

        #print(idx,img_idx)
        if img_idx == 0:
            version = 'gs'
        elif img_idx == 1:
            version = 'hom'
        elif img_idx == 2:
            version = 'sample'
        else:
            version = 'auto'
        # img for this frame
        img = read_img(idx, self.root, 'training', version)
        # annotation for this frame
        K, xyz, scale, verts,mano_list = self.anno_all[idx]
        K, xyz, scale, verts,mano_list = [np.array(x) for x in [K, xyz, scale, verts,mano_list]]
        # crop the image and generate the new K
        handJoint2d = xyz2uvd(xyz, K)[:,:2]
        img_list = [img]
        label2d_list = [handJoint2d]
        img_list, label2d_list, camera = cut_img(img_list, label2d_list, K)
        image_crop = img_list[0]
        # normalize the pose and vert (to millimeter)
        xyz = xyz*1000.0
        verts = verts*1000.0
        root = xyz[9,:]
        xyz_related = (xyz - root)#/(scale*1000.0)
        verts_related = (verts - root)#/(scale*1000.0)
        # data argumentation 
        image_crop, xyz_related, verts_related = processing_augmentation(image_crop, xyz_related, verts_related)
        image_crop = cv2.cvtColor(np.uint8(image_crop), cv2.COLOR_BGR2RGB)
        # adding noise
        image_crop = rgb_processing(image_crop)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))
        # transform K
        K = torch.Tensor(K).float()
        mano_list = torch.Tensor(mano_list).float().view(-1)
        return image_crop, xyz_related, verts_related,{'K':K,'Mano':mano_list}


class FreiHAND_test(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode in ['training', 'evaluation'], 'mode error'
        # load annotations
        self.anno_all = load_db_annotation(root, 'evaluation')
        self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return 3960
        #return 100       
    def __getitem__(self, idx):
        # img for this frame
        img = read_img(idx, self.root, self.mode)
        # annotation for this frame
        K, xyz, scale, verts,mano_list = self.anno_all[idx]
        K, xyz, scale, verts,mano_list = [np.array(x) for x in [K, xyz, scale, verts,mano_list]]
        # crop the image and generate the new K
        handJoint2d = xyz2uvd(xyz, K)[:,:2]
        img_list = [img]
        label2d_list = [handJoint2d]
        img_list, label2d_list, camera = cut_img(img_list, label2d_list, K)
        image_crop = img_list[0]
        # normalize the pose and vert 
        xyz = xyz*1000.0
        verts = verts*1000.0

        #
        pose_uvd = xyz2uvd(xyz, camera)
        vert_uvd = xyz2uvd(verts, camera)
        
        image_crop = cv2.cvtColor(np.uint8(image_crop), cv2.COLOR_BGR2RGB)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))
        #return image_crop, xyz, verts,pose_uvd,vert_uvd
        # transform K
        K = torch.Tensor(K).float()
        mano_list = torch.Tensor(mano_list).float().view(-1)
        return image_crop, xyz, verts,{'K':K,'Mano':mano_list}

inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # path  = "/mnt/data/ziwei/Freihand/"
    path = '/mnt/data/allusers/haipeng/HandData/FreiHAND'
    batch_size = 10

    dataset = FreiHAND(root = path, mode="training")
    trainloader_synthesis = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    dataset = FreiHAND_test(root = path, mode="evaluation")
    test_synthesis = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    for step, (image_crop, xyz, verts,others) in enumerate(test_synthesis):
        batch_image = inv_normalize(image_crop[0])
        image_crop = batch_image.cpu().detach().numpy().transpose(1, 2, 0)

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.imshow(image_crop)

        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.imshow(image_crop)
        # plt.scatter(pose_uvd.cpu().detach().numpy()[0, :, 0],
        #              pose_uvd.cpu().detach().numpy()[0, :, 1],
        #              c='r')


        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.imshow(image_crop)
        # plt.scatter(vert_uvd.cpu().detach().numpy()[0, :, 0],
        #              vert_uvd.cpu().detach().numpy()[0, :, 1],
        #              c='r')

        plt.show()





        

