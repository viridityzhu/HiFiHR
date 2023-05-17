import os
import cv2
import torch
import utils.pytorch_ssim as pytorch_ssim
import lpips
from losses import LossFunction
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func = LossFunction()
lpips_loss = lpips.LPIPS(net="alex").to(device)
# Define the directory where the images are stored
data_dir = "/home/jiayin/S2HAND/outputs/FreiHAND/debug0607/pic/test/per_images"

# Initialize the metrics
PSNR, SSIM, LPIPS, L1, L2 = 0, 0, 0, 0, 0
cnt = 0

# Loop over the files in the data directory
# for filename in os.listdir(data_dir):
for filename in tqdm(os.listdir(data_dir), leave=True):
    if "raw_img" in filename:
        cnt += 1
        # Load the original image
        ori_img = cv2.imread(os.path.join(data_dir, filename))
        # Get the ID of the image
        # img_id = filename.split("_")[0]
        img_id = filename[:12]
        # Load the corresponding predicted image
        pred_img = cv2.imread(os.path.join(data_dir, "{}_re_img.png".format(img_id)))
        # Load the corresponding mask
        mask = cv2.imread(os.path.join(data_dir, "{}_re_sil.png".format(img_id)), cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255
        # Normalize the images
        ori_img = torch.from_numpy(ori_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255
        pred_img = torch.from_numpy(pred_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255
        mask = mask[:,:,720-112:720+112,960-112:960+112]
        ori_img = ori_img[:,:,720-112:720+112,960-112:960+112]
        pred_img = pred_img[:,:,720-112:720+112,960-112:960+112]
        # print(mask.shape)
        # print(ori_img.shape)
        # print(pred_img.shape)
        # exit()
        # Add the data to the list
        mask = mask.to(device)
        ori_img = ori_img.to(device)
        pred_img = pred_img.to(device)

        masked_original_img = ori_img * mask
        masked_pred_img = pred_img * mask
        psnr = -10 * loss_func.MSE_loss(masked_original_img, masked_pred_img).log10().item()
        ssim = pytorch_ssim.ssim(masked_original_img, masked_pred_img).item()
        lpips = lpips_loss(masked_original_img * 2 - 1, masked_pred_img * 2 - 1).mean().item()
        l1 = loss_func.L1_loss(masked_original_img, masked_pred_img).mean().item()
        l2 = loss_func.MSE_loss(masked_original_img, masked_pred_img).mean().item()
        PSNR += psnr
        SSIM += ssim
        LPIPS += lpips
        L1 += l1
        L2 += l2
        print("img{}: {} - PSNR: {:.6f}, SSIM: {:.6f}, LPIPS: {:.6f}, L1: {:.6f}, L2: {:.6f}".format(cnt, img_id,psnr, ssim, lpips, l1, l2))

print("Number of samples: {}".format(cnt))
# Compute the mean of the metrics
num_samples = cnt
mean_PSNR = PSNR / num_samples
mean_SSIM = SSIM / num_samples
mean_LPIPS = LPIPS / num_samples
mean_L1 = L1 / num_samples
mean_L2 = L2 / num_samples

# Print the results
print("mean PSNR: {:.6f}".format(mean_PSNR))
print("mean SSIM: {:.6f}".format(mean_SSIM))
print("mean LPIPS: {:.6f}".format(mean_LPIPS))
print("mean L1: {:.6f}".format(mean_L1))
print("mean L2: {:.6f}".format(mean_L2))
