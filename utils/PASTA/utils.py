import os
import sys

import torch
import torchvision
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from torchvision import transforms as T

class PASTA:
    """
    PASTA: Proportional Amplitude Spectrum Augmentation for Synthetic-to-Real Domain Generalization

    ...

    Attributes
    ----------
    alpha : float
        coefficient of linear term to ensure perturbation strength increases 
        with increasing spatial frequency
    beta : float
        constant perturbation across all frequencies
    k : int
        exponent ensuring non-linear dependence of perturbation on spatial
        frequency

    """
    def __init__(self, alpha: float, beta: float, k: int):
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        fft_src = torch.fft.fftn(img, dim=[-2, -1])
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        X, Y = amp_src.shape[1:]
        X_range, Y_range = None, None

        if X % 2 == 1:
            X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
        else:
            X_range = np.concatenate(
                [np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
            )

        if Y % 2 == 1:
            Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
        else:
            Y_range = np.concatenate(
                [np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
            )

        XX, YY = np.meshgrid(Y_range, X_range)

        exp = self.k
        lin = self.alpha
        offset = self.beta

        inv = np.sqrt(np.square(XX) + np.square(YY))
        inv *= (1 / inv.max()) * lin
        inv = np.power(inv, exp)
        inv = np.tile(inv, (3, 1, 1))
        inv += offset
        prop = np.fft.fftshift(inv, axes=[-2, -1])
        amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)

        aug_img = amp_src * torch.exp(1j * pha_src)
        aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
        aug_img = torch.real(aug_img)
        aug_img = torch.clip(aug_img, 0, 1)
        return aug_img
