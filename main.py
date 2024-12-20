# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import app
from absl import flags
import cv2
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import time
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
#import warnings
#warnings.filterwarnings("ignore")

from nnutils.train_utils import v2s_trainer

opts = flags.FLAGS

def main(_):
    opts.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(opts.local_rank)
    world_size = opts.ngpu
    print("world size:", world_size)
    torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=opts.local_rank,
    )
    print('%d/%d'%(world_size,opts.local_rank))

    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    
    trainer = v2s_trainer(opts)
    data_info = trainer.init_dataset()
    trainer.define_model(data_info)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
