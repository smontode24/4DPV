# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import app
from absl import flags
import cv2
import os.path as osp
import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'.')
import pdb
import time
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import random
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

from nnutils.train_utils import v2s_trainer
import os, re, os.path
opts = flags.FLAGS

def main(_):
    try:
        opts.local_rank = int(os.environ['LOCAL_RANK'])
    except:
        opts.local_rank = 0
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8999"
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(opts.local_rank)
    world_size = 1
    """ torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=1,
        rank=0,
    ) """

    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    random.seed(1)
    
    trainer = v2s_trainer(opts)
    data_info = trainer.init_rec_dataset()    
    trainer.define_model(data_info, finer_step=True)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
