from __future__ import print_function, division
from os import path
import os
from time import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import binary_erosion
from torch.utils.data.sampler import BatchSampler
import random

class FrameBasedDataset(Dataset):
    def __init__(
        self,
        opts,
        root_dir_ds,
        remove_last_elems=False
    ):
        """
        Per-frame loader for finer reconstruction step
        """
        self.opts = opts
        self.img_size = opts.img_size
        self.crop_factor = 1.2 # Default crop factor (area surrounding cat)
        self.flip=0
        self.base_path = root_dir_ds
        self.base_path_imgs = os.path.join(self.base_path, "JPEGImages/Full-Resolution")
        self.base_path_mask = os.path.join(self.base_path, "Annotations/Full-Resolution")
        self.annotations_path = [
            {
                "img_path": img_path, 
                "mask_path": img_path.replace("JPEGImages", "Annotations").replace("jpg", "png"),
                "camera_id": int(img_path.split("/")[-2][len(opts.seqname):]),
                "frame_id": int(img_path.split("/")[-1].split(".")[0])
            } 
            for img_path in sorted(glob.glob(self.base_path_imgs+f"/{opts.seqname}*/*.jpg"))]

        self.cameras = [int(img_path.split("/")[-2][len(opts.seqname):]) for img_path in sorted(glob.glob(self.base_path_imgs+f"/{opts.seqname}*/*.jpg"))]
        self.camera_to_idx = {cam_id: i for i, cam_id in enumerate(np.unique(np.array(self.cameras)))}
        self.offsets = np.array([0]+[np.count_nonzero(np.array(self.cameras) == camera_id) for camera_id in np.unique(np.array(self.cameras))]).cumsum().astype(int)

        if remove_last_elems:
            for i, idx in enumerate(self.offsets[1:]):
                del self.cameras[idx-1-i]
                del self.annotations_path[idx-1-i]

            self.camera_to_idx = {cam_id: i for i, cam_id in enumerate(np.unique(np.array(self.cameras)))}
            self.offsets = np.array([0]+[np.count_nonzero(np.array(self.cameras) == camera_id) for camera_id in np.unique(np.array(self.cameras))]).cumsum().astype(int)


    def __len__(self):
        return len(self.annotations_path)

    def compute_crop_params(self, mask):
        #ss=time.time()
        indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
        center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
        length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
        length = (int(self.crop_factor*length[0]), int(self.crop_factor*length[1]))
        
        #print('center:%f'%(time.time()-ss))

        maxw=self.img_size;maxh=self.img_size
        orisize = (2*length[0], 2*length[1])
        alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
        
        # intrinsics induced by augmentation: augmented to to original img
        # correct cx,cy at clip space (not tx, ty)
        if self.flip==0:
            pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
        else:
            pps  = np.asarray([-float( center[0] - length[0] ), float( center[1] - length[1]  )])
        kaug = np.asarray([alp[0], alp[1], pps[0], pps[1]])

        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        A = np.eye(3)
        B = np.asarray([[alp[0],0,(center[0]-length[0])],
                        [0,alp[1],(center[1]-length[1])],
                        [0,0,1]]).T
        hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
        hp0 = np.dot(hp0,A.dot(B))                   # image coord
        return kaug, hp0, A,B
        
    def __getitem__(self, idx):
        item = self.annotations_path[idx]
        im_path, mask_path = item["img_path"], item["mask_path"]
        try:
            img = cv2.imread(item["img_path"])
            mask = cv2.imread(item["mask_path"], 0)
            camera_id = self.camera_to_idx[item["camera_id"]]
            frame_id = item["frame_id"]
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img[:,:,::-1]
            mask = mask/np.sort(np.unique(mask))[1]
            occluder = mask==255
            mask[occluder] = 0 # What is this?
            if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
                mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
                mask = binary_erosion(mask,iterations=2)

            mask = np.expand_dims(mask, 2)
            kaug, hp0, _, _ = self.compute_crop_params(mask)
            #print('crop params:%f'%(time.time()-ss))
            x0 = hp0[:,:,0].astype(np.float32)
            y0 = hp0[:,:,1].astype(np.float32)
            h, w, _ = img.shape
            img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR)
            mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
            
            img = torch.tensor(img).float().unsqueeze(0).permute(0, 3, 1, 2) / 255
            mask = torch.tensor((mask.astype(np.float32) > 0).astype(np.uint8)).unsqueeze(0)
            unique_id = torch.tensor(np.array([frame_id+self.offsets[camera_id]]))
            camera_id = torch.tensor(np.array([camera_id]).astype(np.int32))
            frame_id = torch.tensor(np.array([frame_id]).astype(np.int32))
            remapping_coords = torch.tensor(hp0).float()
            org_size = torch.tensor([h, w]).float()
            kaug = torch.tensor(kaug).float()

        except Exception as e:
            
            print(
                f"Error processing {im_path} / {mask_path}", e
            )
            return None

        return img, mask, camera_id, frame_id, unique_id, remapping_coords, kaug, org_size

class SameCameraSampler(BatchSampler):
    """
    BatchSampler - Sample images from the same video sequence so that rendering is more efficient  
    """

    def __init__(self, cameras, batch_size, len_dataset):
        self.cameras = cameras
        self.random_gen = random.Random(1) # Mantain selected group consistency across batches
        self.epoch = 0
        self.batch_size = batch_size
        self.len_dataset = len_dataset
        self.labels_set = list(set(self.cameras))
        self.count = 0

    def __iter__(self):
        while self.count * self.batch_size < len(self.dataset):
            # Sample single 
            class_idx = self.random_gen.randint(0, len(self.labels_set)-1)
            video_id = self.labels_set[class_idx]
            indices = self.sample_indices(video_id)
            
            yield indices
            self.count += self.batch_size
        
        self.epoch += 1
        self.count = 0

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def sample_indices(self, video_id):
      """
      Sample indices depending on selected class and epoch.
      """ 
      # replace True in case of a short sequence
      return np.random.choice(np.where(self.cameras == video_id)[0], self.batch_size, replace=True)
