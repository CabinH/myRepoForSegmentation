import torch 
from torch.autograd import Function, Variable
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WeightedBCELoss2d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)

        loss = logits.clamp(min=0)-logits*gt+torch.log(1+torch.exp(-logits.abs()))
        loss = loss*w
        loss = loss.sum() / w.sum()

        return loss

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        #平滑变量
        smooth = 1

        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2*intersection.sum(1)+smooth)/(input_flat.sum(1)+targets_flat.sum(1)+smooth)

        loss = 1 - N_dice_eff.sum()/N

        return loss
        

