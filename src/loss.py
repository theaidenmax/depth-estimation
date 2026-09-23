import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):

  def __init__(self, window_size=11):
    super().__init__()
    self.window_size = window_size

  def forward(self, img1, img2):
    mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size // 2)
    mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size // 2)

    sigma1_sq = (
        F.avg_pool2d(
            img1 * img1, self.window_size, stride=1, padding=self.window_size // 2
        )
        - mu1 * mu1
    )
    sigma2_sq = (
        F.avg_pool2d(
            img2 * img2, self.window_size, stride=1, padding=self.window_size // 2
        )
        - mu2 * mu2
    )
    sigma12 = (
        F.avg_pool2d(
            img1 * img2, self.window_size, stride=1, padding=self.window_size // 2
        )
        - mu1 * mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1 - ssim_map.mean()