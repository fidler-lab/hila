# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

import warnings

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Upscaler(nn.Module):
    def __init__(self, upscale_type, upscale_factor=2, align_corners=False):
        super().__init__()
        self.upscale_type = upscale_type
        self.upscale_factor = upscale_factor
        self.align_corners = align_corners

    def forward(self, x):
        if 'bilinear' in self.upscale_type:
            x = self.resize(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=self.align_corners)
        elif 'nearest' in self.upscale_type:
            x = self.resize(x, scale_factor=self.upscale_factor, mode='nearest', align_corners=self.align_corners)
        return x

    def resize(self, input,
               size=None,
               scale_factor=None,
               mode='nearest',
               align_corners=None):
        if isinstance(size, torch.Size):
            size = tuple(int(x) for x in size)
        return F.interpolate(input, size, scale_factor, mode, align_corners)
