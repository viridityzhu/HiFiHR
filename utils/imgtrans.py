import random
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T

from FDA.utils import FDA_source_to_target_np
from PASTA.utils import PASTA


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img

def add_arm(img, idx):
    if idx >= 32560:
        idx = idx % 32560
    img_id = "%08d" % (idx)
    
    input_image = cv2.imread('/mnt/data/FreiHand/training/rgb/' + img_id + '.jpg')
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    syn_image = np.array(img)
    syn_mask = cv2.imread('/mnt/data/FreiHand_syn/segmentation/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
    syn_mask = (syn_mask > 0).astype(np.uint8) * 255

    syn_hand = cv2.bitwise_and(syn_image, syn_image, mask=syn_mask)

    hand_arm_mask = cv2.imread('/mnt/data/FreiHand_syn/hand_arm_mask/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
    hand_mask = cv2.imread('/mnt/data/FreiHand_syn/hand_mask/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
    arm_mask = hand_arm_mask & ~hand_mask

    hand_with_arm = cv2.bitwise_and(input_image, input_image, mask=arm_mask)
    hand_with_arm_clean = cv2.bitwise_and(hand_with_arm, hand_with_arm, mask=(255 - syn_mask))

    scene = cv2.bitwise_and(syn_image, syn_image, mask = (255 - arm_mask))
    scene = cv2.bitwise_or(scene, scene, mask = (255 - syn_mask))
    
    img_with_arm = scene + hand_with_arm_clean + syn_hand
    
    return img_with_arm

def linear_transform(value, cmin, cmax, a, b):
    return int(a + (b - a) * (value - cmin) / (cmax - cmin))

def add_fourier(img):
    # randomly choose target image
    img_id = str(random.randint(0, 32560 * 4 - 1)).rjust(8, '0')
    # img_id = str(random.randint(0, 3960 - 1)).rjust(8, '0')

    im_src = img
    im_trg = Image.open("/mnt/data/FreiHand/training/rgb/" + img_id + ".jpg").convert('RGB')
    # im_trg = Image.open("/home/zhuoran/data/evaluation/rgb/" + img_id + ".jpg").convert('RGB')

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

    src_in_trg = src_in_trg.transpose((1,2,0))

    cmin, cmax = 0.0, 255.0
    src_in_trg = np.clip((src_in_trg - cmin) * (255.0 / (cmax - cmin)), 0, 255).astype(np.uint8)

    src_in_trg = Image.fromarray(src_in_trg)
    
    return src_in_trg

# def add_pasta_fourier(img):
#     alpha = 3.0
#     beta = 0.25
#     k = 2

#     syn_transform = PASTA(alpha=alpha, beta=beta, k=k)    
#     aug_img = syn_transform(img)

#     return aug_img

def add_pasta(img):
    alpha = 3.0
    beta = 0.25
    k = 2
    
    syn_transform = T.Compose(
        [
            T.ToTensor(),
            PASTA(alpha=alpha, beta=beta, k=k),
            T.ToPILImage(),
        ]
    )
    
    aug_syn_img = syn_transform(img)
    
    return aug_syn_img

def add_occ_obj(img, idx):
    if idx >= 32560:
        idx = idx % 32560
    img_id = "%08d" % (idx)
    
    origin_img = cv2.imread("/mnt/data/FreiHand/training/rgb/" + img_id + ".jpg")
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    final_mask = cv2.imread('/mnt/data/FreiHand_syn/final_mask/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
    final_mask = (final_mask > 0).astype(np.uint8) * 255
    mask_obj = cv2.bitwise_and(origin_img, origin_img, mask=final_mask)
    
    syn_img = np.array(img)
    syn_mask = (final_mask == 0).astype(np.uint8) * 255
    mask_syn_img = cv2.bitwise_and(syn_img, syn_img, mask=syn_mask)
    
    final_syn_img = mask_syn_img + mask_obj
    
    return final_syn_img
