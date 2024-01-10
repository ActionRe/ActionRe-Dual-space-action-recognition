import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.losses import Proxy_Anchor
# from torch import linalg as LA


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.01):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC

# def loss_func(output, fea, z_p, x_kl, label, num_cls, temperature):
#     soft = nn.Softmax(dim=1)
#     label_idx = [i in label for i in range(num_cls)]
#     z_ = torch.stack([fea[label == i].mean(dim=0) for i in range(num_cls)], dim=0)
#     l2 = torch.norm(fea.mean(dim=0))
#     kl_distance = F.mse_loss(z_[label_idx], z_p[label_idx].to(fea.device))
#     kl_loss = F.kl_div(input=soft(z_[label_idx] / temperature),
#                        target=soft(z_p[label_idx].to(fea.device)) / temperature)
#     return kl_distance, l2, kl_loss


def loss_func(z, z_prior, y, num_cls):
    # soft = nn.Softmax(dim=1)
    # smooth = nn.SmoothL1Loss(beta=10.0)
    #label_idx = [i in label for i in range(num_cls)]
    #z_ = torch.stack([fea[label == i].mean(dim=0) for i in range(num_cls)], dim=0)  # 60 256
    #l2 = torch.norm(fea.mean(dim=0))
    #kl_distance = F.mse_loss(z_[label_idx], z_p[label_idx].to(fea.device))
    # kl_loss = F.kl_div(input=soft(z_[label_idx] / temperature),
    #                    target=soft(z_p[label_idx].to(fea.device)) / temperature)
    # p = Proxy_Anchor(nb_classes=num_cls, sz_embed=256, mrg=0.5, alpha=64)
    #ploss = F.smooth_l1_loss(z_[label_idx], z_p[label_idx].to(fea.device))
    #hsic = HSIC(z_[label_idx], z_p[label_idx].to(fea.device), 1, 1)
    # return kl_distance, l2, kl_loss
    #return hsic, l2, ploss
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)  # 60 256
    l2_z_mean = torch.norm(z.mean(dim=0))
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean

