"""
Pytorch Implementation of thr focal loss taken from
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
Credits : https://github.com/clcarwin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
    
    def forward(self, input, target):
       target = target.unsqueeze(1)
       if input.dim()>2:
           input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
           input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
           input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
       target = target.contiguous().view(-1,1)

       logpt = F.log_softmax(input, dim=1)
       logpt = logpt.gather(1,target)
       logpt = logpt.view(-1)
       pt = Variable(logpt.data.exp())

       if self.alpha is not None:
           if self.alpha.type()!=input.data.type():
               self.alpha = self.alpha.type_as(input.data)
           at = self.alpha.gather(0,target.data.view(-1))
           logpt = logpt * Variable(at)

       loss = -1 * (1-pt)**self.gamma * logpt
       if self.reduction == 'mean':
           return loss.mean()
       elif self.reduction == 'sum':
           return loss.sum()
       else:
            return loss