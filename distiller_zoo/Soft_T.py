#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class Softmax_T(nn.Module):
    
    def __init__(self, T):
        super(Softmax_T, self).__init__()
        self.T = T

    def forward(self, y):    
        p = F.softmax(y/self.T, dim=1)
        return p