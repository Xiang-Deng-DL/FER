#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
    
class KL(nn.Module):
    
    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, logit_s, soft_targets):
        
        p_s = F.log_softmax(logit_s/self.T, dim=1)
        
        softmax_loss = - torch.sum(soft_targets * p_s, 1, keepdim=True)
        
        loss = (self.T ** 2) * torch.sum(softmax_loss)

        return loss
    