import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as func_transforms

def get_mask(img_path):
    mask = Image.open(img_path)
    return mask

path = "/home/jiayin/freihand/evaluation/mask/00000000.jpg"
path_right = "/home/jiayin/freihand/training/mask/00000000.jpg"

m1 = get_mask(path)
m2 = get_mask(path_right)
mm1 = torch.round(func_transforms.to_tensor(m1))
mm2 = torch.round(func_transforms.to_tensor(m2))
print(mm1.unique(), mm2.unique())
print(mm1.shape, mm2.shape)