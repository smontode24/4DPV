# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app
import random
import math

import torch
from torch.utils.data import Dataset

import pdb
import glob
from torch.utils.data import DataLoader
import configparser
from utils.io import config_to_dataloader
from dataloader.rec_loader import SameCameraSampler
from torch.utils.data.sampler import BatchSampler

opts = flags.FLAGS
    
def _init_fn(worker_id):
    np.random.seed(1003)
    random.seed(1003)

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts_dict, shuffle=True, train_loader=False):
    # 1*256 workers?!!?!? Makes no sense
    num_workers = opts_dict['n_data_workers'] * opts_dict['batch_size']
    num_workers = min(num_workers, 16)
    # num_workers = 0
    print('# workers: %d'%num_workers)
    print('# pairs: %d'%opts_dict['batch_size'])

    data_inuse = config_to_dataloader(opts_dict)

    sampler = None
    batch_sampler = None
    if opts.consecutive_line_sampler and train_loader:
        if "lineload" in opts_dict.keys():
            batch_sampler = ConsecutiveLineSampler(data_inuse, opts.img_size, opts.n_consecutive, opts.batch_size)

            data_inuse = DataLoader(data_inuse,
                num_workers=num_workers, 
                worker_init_fn=_init_fn, pin_memory=True,
                batch_sampler=batch_sampler)

    if batch_sampler is None:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_inuse,
            num_replicas=opts_dict['ngpu'],
            rank=opts_dict['local_rank'],
            shuffle=True
        )

        data_inuse = DataLoader(data_inuse,
            batch_size= opts_dict['batch_size'], num_workers=num_workers, 
            drop_last=True, worker_init_fn=_init_fn, pin_memory=True,
            sampler=sampler)

    
    return data_inuse

def data_loader_rec(dataset, opts_dict, shuffle=True):
    # 1*256 workers?!!?!? Makes no sense
    num_workers = opts_dict['n_data_workers']
    # num_workers = 0
    print('# workers: %d'%num_workers)
    print('# pairs: %d'%opts_dict['batch_size'])

    if opts_dict['ngpu'] > 1 or True:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=opts_dict['ngpu'],
            rank=opts_dict['local_rank'],
            shuffle=True
        )

        data_inuse = DataLoader(dataset,
         batch_size= opts_dict['batch_size'], num_workers=num_workers, 
         drop_last=True, worker_init_fn=_init_fn, pin_memory=True)
    else:
        sampler = SameCameraSampler(
            dataset.cameras, opts_dict['batch_size'], len(dataset)
        )

        data_inuse = DataLoader(dataset,
            num_workers=num_workers, 
            worker_init_fn=_init_fn, pin_memory=True,
            batch_sampler=sampler)
    return data_inuse

def finer_data_loader(opts_dict, shuffle=True):
    num_workers = opts_dict["n_data_workers"]

#----------- Eval Data Loader ----------#
#----------------------------------#
def eval_loader(opts_dict):
    num_workers = 0
   
    dataset = config_to_dataloader(opts_dict,is_eval=True)
    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=num_workers, drop_last=False, pin_memory=True, shuffle=False)
    return dataset

class ConsecutiveLineSampler(BatchSampler):
    """
    BatchSampler - Samples batches where specific classes are more likely to be sampled as the training runs at each batch.
    At each batch sampling step, one specific class will be selected (e.g., female black), and images of this class will be more likely
    to be sampled. As the epoch number increases,  
    """

    def __init__(self, dataset, img_size, n_consecutive, batch_size):
        self.dataset = dataset
        self.img_size = img_size
        self.n_consecutive = n_consecutive
        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        while True:
            tot = math.ceil(self.batch_size // self.n_consecutive)
            idxs = np.random.randint(0, len(self.dataset), (tot,))
            relative_lines = idxs % self.img_size
            
            too_low = relative_lines < self.n_consecutive + 1
            idxs[too_low] = idxs[too_low] + np.abs(self.n_consecutive - relative_lines[too_low])
            too_high = relative_lines > 512 - self.n_consecutive - 1
            idxs[too_high] = idxs[too_high] - np.abs( self.n_consecutive - (512-relative_lines[too_high]) )
            total_idxs = [np.arange(idx - self.n_consecutive//2, idx + math.ceil(self.n_consecutive/2) ) for idx in idxs[:-1]]
            total_idxs = np.concatenate(total_idxs)

            left_to_sample = self.batch_size - len(total_idxs)
            if left_to_sample > 0:
                total_idxs = np.concatenate([total_idxs, np.arange(idxs[-1] - self.n_consecutive, 
                                idxs[-1] - self.n_consecutive + left_to_sample)])

            yield total_idxs

    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
