# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import numpy as np
import cv2
import torch
from absl import flags
from collections import defaultdict

from nnutils import banmo
from utils.io import extract_data_info
from dataloader import frameloader
from nnutils.geom_utils import get_near_far, sample_xy, K2inv, raycast, chunk_rays
from nnutils.rendering import render_rays
import torch

def image_grid(img, row, col):
    """
    img:     N,h,w,x
    collage: 1,.., x
    """
    bs,h,w,c=img.shape
    device = img.device
    collage = torch.zeros(h*row, w*col, c).to(device)
    for i in range(row):
        for j in range(col):
            collage[i*h:(i+1)*h,j*w:(j+1)*w] = img[i*col+j]
    return collage

def construct_rays_nvs(target_size, rtks, near_far, rndmask, device):
    """
    rndmask: controls which pixel to render
    """
    bs = rtks.shape[0]
    rtks = torch.Tensor(rtks).to(device)
    rndmask = torch.Tensor(rndmask).to(device).view(-1)>0

    img_size = max(target_size)
    _, xys = sample_xy(img_size, bs, 0, device, return_all=True)
    xys=xys.view(img_size,img_size,2)[:target_size[0], :target_size[1]].view(1,-1,2)
    xys = xys[:,rndmask]
    Rmat = rtks[:,:3,:3]
    Tmat = rtks[:,:3,3]
    Kinv = K2inv(rtks[:,3])
    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    return rays