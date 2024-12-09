# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags
import cv2
import time
from tqdm import tqdm
import glob
from copy import deepcopy

import mcubes
from nnutils import banmo
import subprocess
from torch.utils.tensorboard import SummaryWriter
from kmeans_pytorch import kmeans
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import cm
from torch.profiler import profile, record_function, ProfilerActivity
from nnutils import multires_hash
from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam, get_near_far, near_far_to_bound, \
                               compute_point_visibility, process_so3_seq, \
                               ood_check_cse, align_sfm_sim3, gauss_mlp_skinning, \
                               correct_bones, sample_xy, K2inv, raycast, chunk_rays, \
                               subdivide_mesh_py3d, integrated_pos_enc
from nnutils.nerf import grab_xyz_weights, ExtraDFMMLP
from nnutils.rendering import render_rays
from ext_utils.flowlib import flow_to_image
from utils.io import mkdir_p
from nnutils.vis_utils import image_grid, construct_rays_nvs
from dataloader import frameloader
from utils.io import save_vid, draw_cams, extract_data_info, merge_dict,\
        render_root_txt, save_bones, draw_cams_pair, get_vertex_colors
from utils.colors import label_colormap
from dataloader.rec_loader import FrameBasedDataset
import tinycudann as tnn

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    Textures
)

import plotly.graph_objects as go

class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """
    for multi-gpu access
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)

class DPPassthrough(torch.nn.DataParallel):
    """
    for multi-gpu access
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
    
class v2s_trainer():
    def __init__(self, opts, is_eval=False):
        self.opts = opts
        self.is_eval=is_eval
        self.local_rank = opts.local_rank
        if not opts.finer_reconstruction:
            self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        else:
            self.save_dir = os.path.join(opts.checkpoint_dir, opts.frec_exp)
        # self.save_dir = os.path.join(opts.checkpoint_dir, opts.frec_exp)
        self.finer_mesh_step = False
        
        self.accu_steps = opts.accu_steps # Do gradient accumulation, by default to 1, so update weights every iteration
        
        # write logs
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            if not self.is_eval:
                if os.path.exists(log_file):
                    os.remove(log_file)
                opts.append_flags_into_file(log_file)

    def define_model(self, data_info, finer_step=False):
        opts = self.opts
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = banmo.banmo(opts, data_info)
        self.model.forward = self.model.forward_default
        self.num_epochs = opts.num_epochs

        # load model
        if opts.model_path!='':
            self.load_network(opts.model_path, is_eval=self.is_eval)

        if finer_step:
            self.finer_mesh_step = True
            self.define_model_finer_step(data_info)
            self.model.forward = self.model.forward_finer
        #else:
        #    if self.opts.extra_dfm_nr:
        #        if self.opts.dfm_type != "quadratic":
        #            self.model.nerf_models["extra_deform_net"] = ExtraDFMMLP(self.model.in_channels_xyz + opts.t_embed_dim, 
        #                                        n_layers=self.opts.dfm_n_layers, n_hidden_dim=self.opts.n_hidden_dim).to(self.device)
        #        else:
        #            self.model.nerf_models["extra_deform_net"] = ExtraDFMMLP(self.model.in_channels_xyz + opts.t_embed_dim,                     
        #                                        n_layers=self.opts.dfm_n_layers, n_hidden_dim=self.opts.n_hidden_dim, output_dim=3*9).to(self.device)

        if not finer_step:
            if self.is_eval:
                self.model = self.model.to(self.device)
            else:
                # ddp
                self.model = self.model.to(self.device)
                try:
                    self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                    

                    self.model = DataParallelPassthrough(  # torch DataParallel, no modification, just function to set / get attributes
                            self.model,
                            device_ids=[opts.local_rank],
                            output_device=opts.local_rank,
                            find_unused_parameters=True,
                    )
                except:
                    print("no ddp")
        else:
            self.model = self.model.to(self.device)

            self.model = DPPassthrough(  # torch DataParallel, no modification, just function to set / get attributes
                    self.model,
                    device_ids=[opts.local_rank],
                    output_device=opts.local_rank,
            )

        return

    def load_mesh(self):
        # Load coarse mesh extracted from zero-level set of NeRF coarse and convert it to PyTorch3D mesh format        
        # Force object into a mesh through concatenation and merge vertices that share position
        mesh_rest = trimesh.load(self.opts.mesh_path, force="mesh")
        # To PyTorch3d format
        textures = Textures(verts_rgb=torch.tensor(mesh_rest.visual.vertex_colors[:,:3].reshape(1, -1, 3)).float().to("cuda:0")/255.0)
        mesh = Meshes(torch.tensor(mesh_rest.vertices.reshape(1,-1,3)).to(self.device).to(torch.float32), \
            torch.tensor(mesh_rest.faces.reshape(1,-1,3)).to(self.device).to(torch.int64), \
            textures=textures, verts_normals=torch.tensor(mesh_rest.vertex_normals.reshape(1,-1,3)).to(self.device).to(torch.float32))
        #mesh = subdivide_mesh_py3d(mesh, times=self.opts.subdivide_times)
        #print("num vertices:", len(mesh.verts_packed()))
        # Trainable vertices that will be the offset for the mesh -> TODO: What to do when subdividing ?
        deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        verts_rgb = mesh.textures.verts_features_packed().unsqueeze(0)
        verts_rgb.requires_grad = True
        self.model.add_mesh_optimization_parameters(mesh, deform_verts, verts_rgb)

    def define_model_finer_step(self, data_info):
        # Finer step in finer_step
        # Remove unnecessary components: NeRF / uncertainty MLP / ...
        del self.model.nerf_models["coarse"]
        del self.model.nerf_models["nerf_vis"]
        del self.model.nerf_models["nerf_feat"]
        del self.model.nerf_models["nerf_unc"]
        
        # Load mesh and make trainable parts
        self.load_mesh() # Add trainable vertices to optimizer in init_training

        # Initialize renderer | Make trainable lights (based on environment code, put also L1 so that the minimum amount of lights are used)
        # Renderer is a nn module -> add to banmo instance (self.model)
        self.model.initialize_renderer()

        # Additional deformation enforcing local deformations as combination of rotation and translation
        if self.opts.extra_dfm_nr:
            # self.model.nerf_models["time_encoder"] = tnn.Encoding(1, {"n_dims_to_encode": 1, "otype": "Frequency", "n_frequencies": 12})
            #self.model.nerf_models["extra_deform_net"] = ExtraDFMMLP(self.model.in_channels_xyz + self.opts.t_embed_dim).to(self.device)
            #self.model.nerf_models["extra_deform_net"] = tnn.Network(self.model.in_channels_xyz + self.opts.t_embed_dim, 3, multires_hash.nerf_mlp_config).to(self.device)
            self.model.nerf_dfm_mesh_opt = ExtraDFMMLP(self.model.in_channels_xyz + self.opts.t_embed_dim).to(self.device)
            self.model.nerf_models["nerf_dfm_mesh_opt"] = self.model.nerf_dfm_mesh_opt

        if self.opts.trainable_lights:
            num_scenes = data_info["offset"]
            num_pts = self.opts.num_lights
            indices = torch.arange(0, num_pts, dtype=float) + 0.5

            phi = torch.arccos(1 - 2*indices/num_pts)
            theta = torch.pi * (1 + 5**0.5) * indices
            x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
            light_anchor_pos = torch.stack([x, y, z]).transpose(1, 0)
            light_anchor_pos_offsets = torch.zeros_like(light_anchor_pos, requires_grad=True)
            light_anchor_int_a, light_anchor_int_d, light_anchor_int_s = \
                                    torch.zeros_like(light_anchor_pos, requires_grad=True) + 0.5/num_scenes, \
                                    torch.zeros_like(light_anchor_pos, requires_grad=True) + 0.3/num_scenes, \
                                    torch.zeros_like(light_anchor_pos, requires_grad=True) + 0.2/num_scenes 

            self.model.light_anchor_pos = light_anchor_pos
            self.model.light_anchor_pos_offsets = light_anchor_pos_offsets
            self.model.light_anchor_int_a = light_anchor_int_a
            self.model.light_anchor_int_d = light_anchor_int_d
            self.model.light_anchor_int_s = light_anchor_int_s

    def init_rec_dataset(self):
        opts = self.opts
        opts_dict = {}
        opts_dict['n_data_workers'] = opts.n_data_workers
        opts_dict['batch_size'] = opts.batch_size
        opts_dict['seqname'] = opts.seqname
        opts_dict['img_size'] = opts.img_size
        opts_dict['ngpu'] = opts.ngpu
        opts_dict['local_rank'] = opts.local_rank
        opts_dict['rtk_path'] = opts.rtk_path
        opts_dict['preload']= False
        opts_dict['accu_steps'] = opts.accu_steps

        root_data_path = self.opts.data_path
        self.dataloader = FrameBasedDataset(self.opts, root_data_path)
        self.trainloader = FrameBasedDataset(self.opts, root_data_path)
        self.evalloader = FrameBasedDataset(self.opts, root_data_path)

        # Load per frame basis
        self.dataloader = frameloader.data_loader_rec(self.dataloader, opts_dict)
        opts_dict['multiply'] = True
        self.trainloader = frameloader.data_loader_rec(self.trainloader, opts_dict)
        opts_dict['img_size'] = opts.render_size
        self.evalloader = frameloader.data_loader_rec(self.evalloader, opts_dict) # Test dataloader

        return {"offset": self.dataloader.dataset.offsets, "impath": [], "len_evalloader": len(self.evalloader)}

    def init_dataset(self):
        opts = self.opts
        opts_dict = {}
        opts_dict['n_data_workers'] = opts.n_data_workers
        opts_dict['batch_size'] = opts.batch_size
        opts_dict['seqname'] = opts.seqname
        opts_dict['img_size'] = opts.img_size
        opts_dict['ngpu'] = opts.ngpu
        opts_dict['local_rank'] = opts.local_rank
        opts_dict['rtk_path'] = opts.rtk_path
        opts_dict['preload']= False
        opts_dict['accu_steps'] = opts.accu_steps

        if self.is_eval and opts.rtk_path=='' and opts.model_path!='': # In evaluation if cameras have been estimated without given parameters, load them
            # automatically load cameras in the logdir
            model_dir = opts.model_path.rsplit('/',1)[0]
            cam_dir = '%s/init-cam/'%model_dir
            if os.path.isdir(cam_dir):
                opts_dict['rtk_path'] = cam_dir

        self.dataloader = frameloader.data_loader(opts_dict) # Train dataloader
        if opts.lineload:
            opts_dict['lineload'] = True
            opts_dict['multiply'] = True # multiple samples in dataset
            self.trainloader = frameloader.data_loader(opts_dict, train_loader=True)
            opts_dict['lineload'] = False
            del opts_dict['multiply']
        else:
            opts_dict['multiply'] = True
            self.trainloader = frameloader.data_loader(opts_dict, train_loader=True)
            del opts_dict['multiply']
        opts_dict['img_size'] = opts.render_size
        opts_dict['batch_size'] = max(2, opts.batch_size//4)
        self.evalloader = frameloader.eval_loader(opts_dict) # Test dataloader

        # compute data offset
        data_info = extract_data_info(self.evalloader)
        
        return data_info # "offset": Num imgs per video, "impath": path to imgs, "len_evalloader": num items in evaluation data loader
    
    def init_training(self): # TODO: Add here trainable deformation
        opts = self.opts
        # set as module attributes since they do not change across gpus
        if opts.small_training:
            self.model.module.final_steps = self.num_epochs * \
                                    min(opts.num_iters_small_training,len(self.trainloader)) * opts.accu_steps
            # ideally should be greater than 200 batches
        else:
            self.model.module.final_steps = self.num_epochs * len(self.trainloader) 

        params_nerf_coarse=[]
        params_nerf_fine=[]
        params_nerf_beta=[]
        params_nerf_feat=[]
        params_nerf_beta_feat=[]
        #params_nerf_fine=[]
        params_nerf_unc=[]
        params_nerf_flowbw=[]
        params_nerf_skin=[]
        params_nerf_vis=[]
        params_nerf_root_rts=[]
        params_nerf_body_rts=[]
        params_root_code=[]
        params_pose_code=[]
        params_env_code=[]
        params_vid_code=[]
        params_bones=[]
        params_skin_aux=[]
        params_ks=[]
        params_nerf_dp=[]
        params_csenet=[]
        params_mesh_verts=[]
        params_mesh_col=[]
        params_light_env=[]
        params_mesh_cams=[]
        params_mesh_nrd=[]
        params_mesh_lights=[]
        for name,p in self.model.named_parameters():
            if 'nerf_coarse' in name and 'beta' not in name:
                params_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                params_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name:
                params_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                params_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                params_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                params_nerf_unc.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                params_nerf_flowbw.append(p)
            elif 'nerf_skin' in name:
                params_nerf_skin.append(p)
            elif 'nerf_vis' in name:
                params_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                params_nerf_body_rts.append(p)
            elif 'root_code' in name:
                params_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                params_pose_code.append(p)
            elif 'env_code' in name:
                params_env_code.append(p)
            elif 'vid_code' in name:
                params_vid_code.append(p)
            elif 'module.bones' == name:
                params_bones.append(p)
            elif 'module.skin_aux' == name:
                params_skin_aux.append(p)
            elif 'module.ks_param' == name:
                params_ks.append(p)
            elif 'nerf_dp' in name:
                params_nerf_dp.append(p)
            elif 'csenet' in name:
                params_csenet.append(p)
            elif 'env_light_mlp' in name:
                params_light_env.append(p)
            else: continue
            # if opts.local_rank==0:
            #     print('optimized params: %s'%name)

        if opts.finer_reconstruction:
            # TODO: Add camera?
            params_mesh_verts.append(self.model.deform_vertices)
            params_mesh_col.append(self.model.rgb_vertices)

            if opts.optimize_cameras:
                params_mesh_cams.append(self.model.cameras)

            if opts.extra_dfm_nr:
                modules_nr_extra_dfm = [self.model.embedding_xyz, self.model.nerf_dfm_mesh_opt]
                for submodule in modules_nr_extra_dfm:
                    for name, p in submodule.named_parameters():    
                        params_mesh_nrd.append(p)

            if opts.trainable_lights:
                params_mesh_lights = [self.model.light_anchor_pos_offsets, self.model.light_anchor_int_a, self.model.light_anchor_int_d, self.model.light_anchor_int_s]
        else:
            if opts.extra_dfm_nr:
                modules_nr_extra_dfm = [self.model.nerf_models["extra_deform_net"]]
                for submodule in modules_nr_extra_dfm:
                    for name, p in submodule.named_parameters():    
                        params_mesh_nrd.append(p)

        self.optimizer = torch.optim.AdamW(
            [{'params': params_nerf_coarse},
             {'params': params_nerf_beta},
             {'params': params_nerf_feat},
             {'params': params_nerf_beta_feat},
             {'params': params_nerf_fine},
             {'params': params_nerf_unc},
             {'params': params_nerf_flowbw},
             {'params': params_nerf_skin},
             {'params': params_nerf_vis},
             {'params': params_nerf_root_rts},
             {'params': params_nerf_body_rts},
             {'params': params_root_code},
             {'params': params_pose_code},
             {'params': params_env_code},
             {'params': params_vid_code},
             {'params': params_bones},
             {'params': params_skin_aux},
             {'params': params_ks},
             {'params': params_nerf_dp},
             {'params': params_csenet},
             {'params': params_mesh_verts},
             {'params': params_mesh_col},
             {'params': params_light_env},
             {'params': params_mesh_cams},
             {'params': params_mesh_nrd},
             {'params': params_mesh_lights}
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        if self.model.root_basis=='exp':
            lr_nerf_root_rts = 10
        elif self.model.root_basis=='cnn':
            lr_nerf_root_rts = 0.2
        elif self.model.root_basis=='mlp':
            lr_nerf_root_rts = 1 
        elif self.model.root_basis=='expmlp':
            lr_nerf_root_rts = 1 
        else: print('error'); exit()

        last_epoch = -1
        total_steps = opts.total_epochs_all_stages*opts.num_iters_small_training
        try:
            last_epoch = self.model.module.total_steps-1 + opts.current_epochs_total*opts.num_iters_small_training
        except:
            last_epoch = -1

        if opts.reset_lr:
            last_epoch = -1
            total_steps = self.model.module.final_steps

        if opts.one_cycle_lr:

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
                            [opts.learning_rate, # params_nerf_coarse
                            opts.learning_rate, # params_nerf_beta
                            opts.learning_rate, # params_nerf_feat
                        10*opts.learning_rate, # params_nerf_beta_feat
                            opts.learning_rate, # params_nerf_fine
                            opts.learning_rate, # params_nerf_unc
                            opts.learning_rate, # params_nerf_flowbw
                            opts.learning_rate, # params_nerf_skin
                            opts.learning_rate, # params_nerf_vis
            lr_nerf_root_rts*opts.learning_rate, # params_nerf_root_rts
                            opts.learning_rate, # params_nerf_body_rts
            lr_nerf_root_rts*opts.learning_rate, # params_root_code
                            opts.learning_rate, # params_pose_code
                            opts.learning_rate, # params_env_code
                            opts.learning_rate, # params_vid_code
                            opts.learning_rate, # params_bones
                        10*opts.learning_rate, # params_skin_aux
                        10*opts.learning_rate, # params_ks
                            opts.learning_rate, # params_nerf_dp
                            opts.learning_rate, # params_csenet
        opts.lr_offsets_c*opts.learning_rate, # params_mesh_verts
        opts.lr_offsets_col*opts.learning_rate, # params_mesh_col
                            opts.learning_rate, # params_light_env
        opts.lr_offsets_cams*opts.learning_rate, # params_mesh_cams
        opts.lr_offsets_d*opts.learning_rate, # params_mesh_nrd
                            opts.learning_rate  # params_mesh_lights
                ],
                int(total_steps/self.accu_steps),
                pct_start=2./self.num_epochs, # use 2 epochs to warm up
                cycle_momentum=False, 
                anneal_strategy='linear',
                final_div_factor=1./5, div_factor = 25
                )
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                    (opts.min_lr_expdecay/opts.max_lr_expdecay) ** (1/total_steps) # go from 5e-4 -> 1e-5
                    ) 

        if last_epoch != -1:
            for _ in range(last_epoch):
                self.scheduler.step() # Set to current LR, not best way but I don't know an easier way
        
        

    def save_network(self, epoch_label, prefix=''):
        if self.opts.local_rank==0:
            param_path = '%s/%sparams_%s.pth'%(self.save_dir,prefix,epoch_label)
            save_dict = self.model.state_dict()
            torch.save(save_dict, param_path)

            var_path = '%s/%svars_%s.npy'%(self.save_dir,prefix,epoch_label)
            latest_vars = self.model.latest_vars.copy()
            del latest_vars['fp_err']  
            del latest_vars['flo_err']   
            del latest_vars['sil_err'] 
            del latest_vars['flo_err_hist']
            np.save(var_path, latest_vars)
            return
    
    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
            new_dict[i] = v
        return new_dict

    def load_network(self,model_path=None, is_eval=True, rm_prefix=True):
        opts = self.opts
        states = torch.load(model_path,map_location='cpu')
        if rm_prefix: states = self.rm_module_prefix(states)
        var_path = model_path.replace('params', 'vars').replace('.pth', '.npy')
        latest_vars = np.load(var_path,allow_pickle=True)[()]
        
        if is_eval:
            # load variables
            self.model.latest_vars = latest_vars
        
        if not opts.finer_reconstruction:
            # if size mismatch, delete all related variables
            if rm_prefix and states['near_far'].shape[0] != self.model.near_far.shape[0]:
                print('!!!deleting video specific dicts due to size mismatch!!!')
                self.del_key( states, 'near_far') 
                self.del_key( states, 'root_code.weight') # only applies to root_basis=mlp
                self.del_key( states, 'pose_code.weight')
                self.del_key( states, 'pose_code.basis_mlp.weight')
                self.del_key( states, 'nerf_body_rts.0.weight')
                self.del_key( states, 'nerf_body_rts.0.basis_mlp.weight')
                self.del_key( states, 'nerf_root_rts.0.weight')
                self.del_key( states, 'nerf_root_rts.root_code.weight')
                self.del_key( states, 'nerf_root_rts.root_code.basis_mlp.weight')
                self.del_key( states, 'nerf_root_rts.delta_rt.0.basis_mlp.weight')
                self.del_key( states, 'nerf_root_rts.base_rt.se3')
                self.del_key( states, 'nerf_root_rts.delta_rt.0.weight')
                self.del_key( states, 'env_code.weight')
                self.del_key( states, 'env_code.basis_mlp.weight')
                if 'vid_code.weight' in states.keys():
                    self.del_key( states, 'vid_code.weight')
                if 'ks_param' in states.keys():
                    self.del_key( states, 'ks_param')

                # delete pose basis(backbones)
                if not opts.keep_pose_basis:
                    del_key_list = []
                    for k in states.keys():
                        if 'nerf_body_rts' in k or 'nerf_root_rts' in k:
                            del_key_list.append(k)
                    for k in del_key_list:
                        print(k)
                        self.del_key( states, k)
        
            if rm_prefix and opts.lbs and states['bones'].shape[0] != self.model.bones.shape[0]:
                self.del_key(states, 'bones')
                states = self.rm_module_prefix(states, prefix='nerf_skin')
                states = self.rm_module_prefix(states, prefix='nerf_body_rts')


            # load some variables
            # this is important for volume matching
            if latest_vars['obj_bound'].size==1:
                latest_vars['obj_bound'] = latest_vars['obj_bound'] * np.ones(3)
            self.model.latest_vars['obj_bound'] = latest_vars['obj_bound'] 

            # load nerf_coarse, nerf_bone/root (not code), nerf_vis, nerf_feat, nerf_unc
            #TODO somehow, this will reset the batch stats for 
            # a pretrained cse model, to keep those, we want to manually copy to states
            if opts.ft_cse and \
            'csenet.net.backbone.fpn_lateral2.weight' not in states.keys():
                self.add_cse_to_states(self.model, states)
        
        if opts.fr_ckpt and opts.finer_reconstruction:
            # Define cameras
            # sorted(glob.glob(self.base_path_imgs+f"/{opts.seqname}*/*.jpg"))
            self.model.cameras = torch.stack([torch.tensor(np.loadtxt(cam_path)) for cam_path in \
                                sorted(glob.glob("/".join(opts.model_path.split("/")[:-1])+"/*-cam-*"))]).float()
            self.model.cameras.requires_grad = opts.optimize_cameras

        self.model.load_state_dict(states, strict=False)

        """ dfm_code_loaded = False
        for k in list(states.keys()):
            if "dfm_code" in k:
                dfm_code_loaded = True
        
        if not dfm_code_loaded:
            self.model.dfm_code = self.model.pose_code.clone().detach() """

        if opts.finer_reconstruction:
            opts = self.opts
            opts_dict = {}
            opts_dict['n_data_workers'] = opts.n_data_workers
            opts_dict['batch_size'] = opts.batch_size
            opts_dict['seqname'] = opts.seqname
            opts_dict['img_size'] = opts.img_size
            opts_dict['ngpu'] = opts.ngpu
            opts_dict['local_rank'] = opts.local_rank
            opts_dict['rtk_path'] = opts.rtk_path
            opts_dict['preload']= False
            opts_dict['accu_steps'] = opts.accu_steps

            root_data_path = self.opts.data_path
            self.dataloader = FrameBasedDataset(self.opts, root_data_path, remove_last_elems=True)
            self.trainloader = FrameBasedDataset(self.opts, root_data_path, remove_last_elems=True)
            self.evalloader = FrameBasedDataset(self.opts, root_data_path, remove_last_elems=True)

            # Load per frame basis
            self.dataloader = frameloader.data_loader_rec(self.dataloader, opts_dict)
            opts_dict['multiply'] = True
            self.trainloader = frameloader.data_loader_rec(self.trainloader, opts_dict)
            opts_dict['img_size'] = opts.render_size
            self.evalloader = frameloader.data_loader_rec(self.evalloader, opts_dict)
            self.model.data_offset = self.dataloader.dataset.offsets

        if not opts.fr_ckpt and opts.finer_reconstruction:
            self.model.cameras = torch.stack([torch.tensor(np.loadtxt(cam_path)) for cam_path in \
                                            sorted(glob.glob("/".join(opts.model_path.split("/")[:-1])+"/*-cam-*"))]).float()
            self.model.cameras.requires_grad = opts.optimize_cameras

        if opts.finer_reconstruction:
            self.model.cameras.to(self.device)
        
        return

    @staticmethod 
    def add_cse_to_states(model, states):
        states_init = model.state_dict()
        for k in states_init.keys():
            v = states_init[k]
            if 'csenet' in k:
                states[k] = v

    def eval_cam(self, idx_render=None): 
        """
        idx_render: list of frame index to render
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()
            # load data
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = False
            batch = []
            for i in idx_render:
                batch.append( self.evalloader.dataset[i] )
            batch = self.evalloader.collate_fn(batch)
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = True

            #TODO can be further accelerated
            self.model.convert_batch_input(batch)

            if opts.unc_filter:
                # process densepoe feature # What is valid list / error list?
                valid_list, error_list = ood_check_cse(self.model.dp_feats, 
                                        self.model.dp_embed, 
                                        self.model.dps.long()) # dps contain idx of which vertex does each feature correspond to in the 3d mesh vertices?
                valid_list = valid_list.cpu().numpy()
                error_list = error_list.cpu().numpy()
            else:
                valid_list = np.ones( len(idx_render))
                error_list = np.zeros(len(idx_render))

            self.model.convert_root_pose()
            rtk = self.model.rtk
            kaug = self.model.kaug

            #TODO may need to recompute after removing the invalid predictions
            # need to keep this to compute near-far planes
            self.model.save_latest_vars()
                
            # extract mesh sequences
            aux_seq = {
                       'is_valid':[],
                       'err_valid':[],
                       'rtk':[],
                       'kaug':[],
                       'impath':[],
                       'masks':[],
                       }
            for idx,_ in enumerate(idx_render):
                frameid=self.model.frameid[idx]
                #if opts.local_rank==0: 
                #    print('extracting frame %d'%(frameid.cpu().numpy()))
                aux_seq['rtk'].append(rtk[idx].cpu().numpy())
                aux_seq['kaug'].append(kaug[idx].cpu().numpy())
                aux_seq['masks'].append(self.model.masks[idx].cpu().numpy())
                aux_seq['is_valid'].append(valid_list[idx])
                aux_seq['err_valid'].append(error_list[idx])
                
                impath = self.model.impath[frameid.long()]
                aux_seq['impath'].append(impath)
        return aux_seq
  
    def eval(self, idx_render=None, dynamic_mesh=False, keep_original_mesh=False): 
        """
        idx_render: list of frame index to render
        dynamic_mesh: whether to extract canonical shape, or dynamic shape
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()

            bounds = self.model.latest_vars['obj_bound']
            if opts.bound_fine and opts.fine_nerf_net:
                bounds = bounds*opts.bound_factor_fine

            # run marching cubes on canonical shape
            # fine_nerf_net 🧐
            mesh_dict_rest = self.extract_mesh(self.model, opts.chunk, \
                                         opts.sample_grid3d, opts.mc_threshold, \
                                         keep_original_mesh=keep_original_mesh, bound = bounds)

            # choose a grid image or the whold video
            if idx_render is None: # render 9 frames
                idx_render = np.linspace(0,len(self.evalloader)-1, 9, dtype=int)

            # render mesh, with skinning weights, etc. for 9 frames or more (idx_render param)
            chunk=opts.rnd_frame_chunk
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'impath':[],
                       'bone':[],
                       'view_dir':[]}

            for j in range(0, len(idx_render), chunk):
                batch = []
                idx_chunk = idx_render[j:j+chunk]
                for i in idx_chunk:
                    batch.append( self.evalloader.dataset[i] )
                batch = self.evalloader.collate_fn(batch)
                
                #try:
                rendered = self.render_vid(self.model, batch) 
            
                for k, v in rendered.items():
                    rendered_seq[k] += [v]
                    
                hbs=len(idx_chunk)
                sil_rszd = F.interpolate(self.model.masks[:hbs,None], 
                            (opts.render_size, opts.render_size))[:,0,...,None]
                rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:hbs]]
                rendered_seq['sil'] += [self.model.masks[...,None]      [:hbs]]
                rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:hbs]]
                rendered_seq['dpc'] += [self.model.dp_vis[self.model.dps.long()][:hbs]]
                rendered_seq['occ'] += [self.model.occ[...,None]      [:hbs]]
                rendered_seq['feat']+= [self.model.dp_feats.std(1)[...,None][:hbs]]
                rendered_seq['flo_coarse'][-1]       *= sil_rszd 
                rendered_seq['img_loss_samp'][-1]    *= sil_rszd 
                if opts.dist_corresp:
                    if 'frame_cyc_dis' in rendered_seq.keys() and \
                        len(rendered_seq['frame_cyc_dis'])>0:
                        rendered_seq['frame_cyc_dis'][-1] *= 255/rendered_seq['frame_cyc_dis'][-1].max()
                        rendered_seq['frame_rigloss'][-1] *= 255/rendered_seq['frame_rigloss'][-1].max()
                if opts.use_embed:
                    rendered_seq['pts_pred'][-1] *= sil_rszd 
                    rendered_seq['pts_exp'] [-1] *= rendered_seq['sil_coarse'][-1]
                    rendered_seq['feat_err'][-1] *= sil_rszd
                    rendered_seq['feat_err'][-1] *= 255/rendered_seq['feat_err'][-1].max()
                if opts.use_proj:
                    rendered_seq['proj_err'][-1] *= sil_rszd
                    rendered_seq['proj_err'][-1] *= 255/rendered_seq['proj_err'][-1].max()
                if opts.use_unc or opts.train_always_unc:
                    rendered_seq['unc_pred'][-1] -= rendered_seq['unc_pred'][-1].min()
                    rendered_seq['unc_pred'][-1] *= 255/rendered_seq['unc_pred'][-1].max()
                #except:
                #    print("Could not render video")
                    
                # extract mesh sequences
                for idx in range(len(idx_chunk)):
                    frameid=self.model.frameid[idx].long()
                    embedid=self.model.embedid[idx].long()
                    # print('extracting frame %d'%(frameid.cpu().numpy()))
                    # run marching cubes
                    if dynamic_mesh:
                        if not opts.queryfw:
                           mesh_dict_rest=None 
                        mesh_dict = self.extract_mesh(self.model,opts.chunk,
                                            opts.sample_grid3d, opts.mc_threshold,
                                        embedid=embedid, mesh_dict_in=mesh_dict_rest, bound=bounds)
                        mesh=mesh_dict['mesh']
                        #if not opts.ce_color:
                        # get view direction 
                        obj_center = self.model.rtk[idx][:3,3:4]
                        cam_center = -self.model.rtk[idx][:3,:3].T.matmul(obj_center)[:,0]
                        view_dir = torch.cuda.FloatTensor(mesh.vertices, device=self.device) \
                                        - cam_center[None]
                        vis = get_vertex_colors(self.model, mesh, 
                                                frame_idx=idx, view_dir=view_dir, fine_nerf_net=opts.fine_nerf_net)
                        mesh.visual.vertex_colors[:,:3] = vis*255
                        aux_seq['view_dir'].append(view_dir)
                        
                        # save bones
                        if 'bones' in mesh_dict.keys():
                            bone = mesh_dict['bones'][0].cpu().numpy()
                            aux_seq['bone'].append(bone)
                    else:
                        mesh=mesh_dict_rest['mesh']
                    aux_seq['mesh'].append(mesh)

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[idx].cpu().numpy())
                    
                    # save image list
                    impath = self.model.impath[frameid]
                    aux_seq['impath'].append(impath)

            # save canonical mesh and extract skinning weights
            mesh_rest = aux_seq['mesh_rest']
            if len(mesh_rest.vertices)>100:
                self.model.latest_vars['mesh_rest'] = mesh_rest
            if opts.lbs:
                bones_rst = self.model.bones
                bones_rst,_ = correct_bones(self.model, bones_rst)
                # compute skinning color
                if mesh_rest.vertices.shape[0]>100:
                    rest_verts = torch.Tensor(mesh_rest.vertices).to(self.device)
                    nerf_skin = self.model.nerf_skin if opts.nerf_skin else None
                    rest_pose_code = self.model.rest_pose_code(torch.Tensor([0])\
                                            .long().to(self.device))
                    skins = gauss_mlp_skinning(rest_verts[None], 
                            self.model.embedding_xyz,
                            bones_rst, rest_pose_code, 
                            nerf_skin, skin_aux=self.model.skin_aux)[0]
                    skins = skins.cpu().numpy()
   
                    num_bones = skins.shape[-1]
                    colormap = label_colormap()
                    # TODO use a larger color map
                    colormap = np.repeat(colormap[None],4,axis=0).reshape(-1,3)
                    colormap = colormap[:num_bones]
                    colormap = (colormap[None] * skins[...,None]).sum(1)

                    mesh_rest_skin = mesh_rest.copy()
                    mesh_rest_skin.visual.vertex_colors = colormap
                    aux_seq['mesh_rest_skin'] = mesh_rest_skin

                aux_seq['bone_rest'] = bones_rst.cpu().numpy()
        
            # draw camera trajectory
            suffix_id=0
            if hasattr(self.model, 'epoch'):
                suffix_id = self.model.epoch
            if opts.local_rank==0:
                mesh_cam = draw_cams(aux_seq['rtk'])
                mesh_cam.export('%s/mesh_cam-%02d.obj'%(self.save_dir,suffix_id))
            
                mesh_path = '%s/mesh_rest-%02d.obj'%(self.save_dir,suffix_id)
                mesh_rest.export(mesh_path)

                if opts.lbs:
                    bone_rest = aux_seq['bone_rest']
                    bone_path = '%s/bone_rest-%02d.obj'%(self.save_dir,suffix_id)
                    save_bones(bone_rest, 0.1, bone_path)

            try:
                # save images
                for k in list(rendered_seq.keys()):
                    rendered_seq[k] = torch.cat(rendered_seq[k],0)
            except:
                print("no images to render")
                #if opts.local_rank==0:
                #    is_flow = self.isflow(k)
                #    upsample_frame = min(1,len(rendered_seq[k]))
                #    save_vid('%s/%s'%(self.save_dir,k), 
                #            rendered_item.cpu().numpy(), 
                #            suffix='.gif', upsample_frame=upsample_frame, 
                #            is_flow=is_flow)

        return rendered_seq, aux_seq

    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            if not opts.finer_reconstruction:
                log = SummaryWriter(os.path.join(opts.checkpoint_dir,opts.logname), comment=opts.logname)
            else:
                log = SummaryWriter(os.path.join(opts.checkpoint_dir,opts.frec_exp), comment=opts.logname)
        else: log=None
        self.model.module.total_steps = 0
        self.model.module.progress = 0
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)

        if not opts.finer_reconstruction: # NR-NeRF stage
            # disable bones before warmup epochs are finished
            if opts.lbs: 
                self.model.num_bone_used = 0
                del self.model.module.nerf_models['bones']
            if opts.lbs and opts.nerf_skin:
                del self.model.module.nerf_models['nerf_skin']
        
            # warmup shape # Initialize NeRF coarse so that it outputs the SDF modeling a sphere / ellipsis (sigma = SDF here)
            if opts.warmup_shape_ep>0:
                print("[I] Warmup shape to sphere / ellipsis")
                self.warmup_shape(log)

            # CNN pose warmup or load CNN
            if opts.warmup_pose_ep>0 or opts.pose_cnn_path!='':
                self.warmup_pose(log, pose_cnn_path=opts.pose_cnn_path) # Warm up pose ResNet18 (if needed) & Extract camera parameters estimation with PoseNet 
            else:
                # save cameras to latest vars and file
                if opts.use_rtk_file:
                    self.model.module.use_cam=True
                    self.extract_cams(self.dataloader)
                    self.model.module.use_cam=opts.use_cam
                else:
                    self.extract_cams(self.dataloader)

            #TODO train mlp
            if opts.warmup_rootmlp: # Set rotation matrix as quaternion angles
                # set se3 directly # https://www.seas.upenn.edu/~meam620/slides/kinematicsI.pdf Rigid body transformation (R + t)
                rmat = torch.Tensor(self.model.latest_vars['rtk'][:,:3,:3])
                quat = transforms.matrix_to_quaternion(rmat).to(self.device)
                self.model.module.nerf_root_rts.base_rt.se3.data[:,3:] = quat # Initial matrix parameters estimation without the delta modification

            # clear buffers for pytorch1.10+
            try: self.model._assign_modules_buffers()
            except: pass
            
            # set near-far plane
            if opts.model_path=='':
                self.reset_nf()

            # reset idk in latest_vars
            self.model.module.latest_vars['idk'][:] = 0. # What is idk? I don't know or something? About what?
    
            #TODO save loaded wts of posecs
            if opts.freeze_coarse: # In first optimization this is set to false
                self.model.module.shape_xyz_wt = \
                    grab_xyz_weights(self.model.module.nerf_coarse, clone=True)
                self.model.module.skin_xyz_wt = \
                    grab_xyz_weights(self.model.module.nerf_skin, clone=True)
                self.model.module.feat_xyz_wt = \
                    grab_xyz_weights(self.model.module.nerf_feat, clone=True)

            #TODO reset beta
            if opts.reset_beta: # Beta of NeRF
                self.model.module.nerf_coarse.beta.data[:] = 0.1
                if opts.fine_nerf_net:
                    self.model.module.nerf_fine.beta.data[:] = 0.1

        eval_every_n = opts.eval_every_n

        # start training
        for epoch in range(0, self.num_epochs):
            self.model.epoch = epoch

            torch.cuda.empty_cache()
            self.model.module.img_size = opts.img_size
            if epoch==0: self.save_network('0') # to save some cameras
            if False: self.add_image_grid(rendered_seq, log, epoch) # if #opts.local_rank==0: # No idea (I think is for the figure of the webpage)
            # Reset some parameters like reinitializing bones if necessary. Also reset logs from last epoch.
            self.reset_hparams(epoch)
            
            torch.cuda.empty_cache()
            
            ## TODO harded coded
            #if opts.freeze_proj:
            #    if self.model.module.progress<0.8:
            #        #opts.nsample=64
            #        opts.ndepth=2
            #    else:
            #        #opts.nsample = nsample
            #        opts.ndepth = self.model.module.ndepth_bk
            if not opts.finer_reconstruction:
                self.train_one_epoch(epoch, log, track_progress=True)
                try:
                    if opts.local_rank==0:
                        if (epoch % eval_every_n == 0 or not opts.small_training) and epoch > 0:
                            print("Generating synthetic views from training samples...")
                            self.model.module.img_size = opts.render_size
                            rendered_seq, aux_seq = self.eval()
                            # w_extra_dfm=True, frames_to_render=20, extra_suffix=""
                            self.debug_view_synthesis_progress(epoch, self.model.module.total_steps // len(self.trainloader), \
                                w_extra_dfm=False, frames_to_render=20, extra_suffix="nodfm")
                            if opts.extra_dfm_nr and ppts.debug_detail:
                                self.debug_view_synthesis_progress(epoch, self.model.module.total_steps // len(self.trainloader), \
                                    w_extra_dfm=True, frames_to_render=20, extra_suffix="dfm")
                                self.debug_vector_field(epoch, self.model.module.total_steps // len(self.trainloader))
                            self.model.module.img_size = opts.img_size
                except:
                    print("Failed to debug views")
            else:
                self.train_one_epoch_rec(epoch, log, track_progress=True)
            
            # print('saving the model at the end of epoch {:d}, iters {:d}'.\
            #                   format(epoch, self.model.module.total_steps))
            self.save_network('latest')
            #self.save_network(str(epoch+1)) # Don't save each epoch...

    @staticmethod
    def save_cams(opts,aux_seq, save_prefix, latest_vars,datasets, evalsets, obj_scale,
            trainloader=None, unc_filter=True):
        """
        save cameras to dir and modify dataset 
        """
        mkdir_p(save_prefix)
        dataset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in datasets}
        evalset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in evalsets}
        if trainloader is not None:
            line_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in trainloader}

        length = len(aux_seq['impath'])
        valid_ids = aux_seq['is_valid']
        idx_combine = 0
        for i in range(length):
            impath = aux_seq['impath'][i]
            seqname = impath.split('/')[-2]
            rtk = aux_seq['rtk'][i]
           
            if unc_filter:
                # in the same sequance find the closest valid frame and replace it
                seq_idx = np.asarray([seqname == i.split('/')[-2] \
                        for i in aux_seq['impath']])
                valid_ids_seq = np.where(valid_ids * seq_idx)[0]
                if opts.local_rank==0 and i==0: 
                    print('%s: %d frames are valid'%(seqname, len(valid_ids_seq)))
                if len(valid_ids_seq)>0 and not aux_seq['is_valid'][i]:
                    closest_valid_idx = valid_ids_seq[np.abs(i-valid_ids_seq).argmin()]
                    rtk[:3,:3] = aux_seq['rtk'][closest_valid_idx][:3,:3]

            # rescale translation according to input near-far plane
            rtk[:3,3] = rtk[:3,3]*obj_scale
            rtklist = dataset_dict[seqname].rtklist
            idx = int(impath.split('/')[-1].split('.')[-2])
            save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx)
            np.savetxt(save_path, rtk)
            rtklist[idx] = save_path
            evalset_dict[seqname].rtklist[idx] = save_path
            if trainloader is not None:
                line_dict[seqname].rtklist[idx] = save_path
            
            #save to rtraw 
            latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
            latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]

            if idx==len(rtklist)-2:
                # to cover the last
                save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx+1)
                if opts.local_rank==0: print('writing cam %s'%save_path)
                np.savetxt(save_path, rtk)
                rtklist[idx+1] = save_path
                evalset_dict[seqname].rtklist[idx+1] = save_path
                if trainloader is not None:
                    line_dict[seqname].rtklist[idx+1] = save_path

                idx_combine += 1
                latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
                latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]
            idx_combine += 1
        
    #def extract_cameras_from_previous_steps(self):
    #    # Set camera variables here -> Allow optimization of camera pose?
    #    base_path_cameras = "/".join(self.opts.model_path.split("/")[:-1])
    #    cam_paths = sorted(glob.glob(base_path_cameras+"/*-cam-*.txt"))
    #    self.cameras = {cam_path.split("/")[-1]: torch.tensor(np.loadtxt(cam_path).astype(np.float32), requires_grad=True) for cam_path in cam_paths}
        
    def extract_cams(self, full_loader):
        # store cameras
        opts = self.opts
        idx_render = range(len(self.evalloader))
        chunk = 50
        aux_seq = []
        # TODO: Load initial cameras if already computed? From same cat-pikachiu independently from the experiment
        eval_cam_bar = tqdm(range(0, len(idx_render)))       
        for i in range(0, len(idx_render), chunk):
            aux_seq.append(self.eval_cam(idx_render=idx_render[i:i+chunk]))
            eval_cam_bar.update(chunk)
            eval_cam_bar.set_description(
                f"Extracting initial parameters from PoseNet {i+1}/{len(idx_render)}"
            )

        aux_seq = merge_dict(aux_seq)
        aux_seq['rtk'] = np.asarray(aux_seq['rtk'])
        aux_seq['kaug'] = np.asarray(aux_seq['kaug'])
        aux_seq['masks'] = np.asarray(aux_seq['masks'])
        aux_seq['is_valid'] = np.asarray(aux_seq['is_valid'])
        aux_seq['err_valid'] = np.asarray(aux_seq['err_valid'])

        save_prefix = '%s/init-cam'%(self.save_dir)
        trainloader=self.trainloader.dataset.datasets
        self.save_cams(opts,aux_seq, save_prefix,
                    self.model.module.latest_vars,
                    full_loader.dataset.datasets,
                self.evalloader.dataset.datasets,
                self.model.obj_scale, trainloader=trainloader,
                unc_filter=opts.unc_filter)
        
        dist.barrier() # wait untail all have finished
        if opts.local_rank==0:
            # draw camera trajectory
            for dataset in full_loader.dataset.datasets:
                seqname = dataset.imglist[0].split('/')[-2]
                render_root_txt('%s/%s-'%(save_prefix,seqname), 0)


    def reset_nf(self): # Reset near-far plane by moving vertices and save near-far plane from current mesh
        opts = self.opts
        # save near-far plane
        shape_verts = self.model.dp_verts_unit / 3 * self.model.near_far.mean()
        shape_verts = shape_verts * 1.2
        # save object bound if first stage
        if opts.model_path=='' and opts.bound_factor>0:
            shape_verts = shape_verts*opts.bound_factor
            self.model.module.latest_vars['obj_bound'] = \
            shape_verts.abs().max(0)[0].detach().cpu().numpy()

        if self.model.near_far[:,0].sum()==0: # if no valid nf plane loaded
            self.model.near_far.data = get_near_far(self.model.near_far.data,
                                                self.model.latest_vars,
                                         pts=shape_verts.detach().cpu().numpy())
        save_path = '%s/init-nf.txt'%(self.save_dir)
        save_nf = self.model.near_far.data.cpu().numpy() * self.model.obj_scale
        np.savetxt(save_path, save_nf)
    
    def warmup_shape(self, log):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.module.forward = self.model.module.forward_warmup_shape # Warmup of shape
        full_loader = self.trainloader  # store original loader
        self.trainloader = range(200)
        self.num_epochs = opts.warmup_shape_ep

        # training
        self.init_training()
        tr_progress_bar = tqdm(range(0, opts.warmup_shape_ep))
        for epoch in tr_progress_bar:
            self.model.epoch = epoch
            self.train_one_epoch(epoch, log, warmup=True)
            self.save_network(str(epoch+1), 'mlp-') 
            tr_progress_bar.update(1)
            tr_progress_bar.set_description(
                f"Warmup shape epoch {epoch+1}/{opts.warmup_shape_ep}"
            )

        # restore dataloader, rts, forward function
        self.model.module.forward = self.model.module.forward_default
        self.trainloader = full_loader
        self.num_epochs = opts.num_epochs

        # start from low learning rate again
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.

    def warmup_pose(self, log, pose_cnn_path):
        opts = self.opts # Warmup ResNet18 for predicting rotation matrix (Appendix B.1)

        # force using warmup forward, dataloader, cnn root
        self.model.module.root_basis = 'cnn'
        self.model.module.use_cam = False
        self.model.module.forward = self.model.module.forward_warmup
        full_loader = self.dataloader  # store original loader
        self.dataloader = range(200)
        original_rp = self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = self.model.module.dp_root_rts
        del self.model.module.dp_root_rts
        self.num_epochs = opts.warmup_pose_ep
        self.model.module.is_warmup_pose=True

        if pose_cnn_path=='': # Load PoseNet ResNet18, train if pretrained PoseNet model not available
            # training
            self.init_training()
            for epoch in range(0, opts.warmup_pose_ep):
                self.model.epoch = epoch
                self.train_one_epoch(epoch, log, warmup=True)
                self.save_network(str(epoch+1), 'cnn-') 

                # eval
                #_,_ = self.model.forward_warmup(None)
                # rendered_seq = self.model.warmup_rendered 
                # if opts.local_rank==0: self.add_image_grid(rendered_seq, log, epoch)
        else: 
            pose_states = torch.load(opts.pose_cnn_path, map_location='cpu')
            pose_states = self.rm_module_prefix(pose_states, 
                    prefix='module.nerf_root_rts')
            self.model.module.nerf_root_rts.load_state_dict(pose_states, 
                                                        strict=False)

        # extract camera and near far planes
        self.extract_cams(full_loader)

        # restore dataloader, rts, forward function
        self.model.module.root_basis=opts.root_basis
        self.model.module.use_cam = opts.use_cam
        self.model.module.forward = self.model.module.forward_default
        self.dataloader = full_loader
        del self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = original_rp
        self.num_epochs = opts.num_epochs
        self.model.module.is_warmup_pose=False

        # start from low learning rate again
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.
            
    def train_one_epoch(self, epoch, log, warmup=False, track_progress=False):
        """
        training loop in a epoch
        """
        opts = self.opts
        self.model.train()
        dataloader = self.trainloader

        if not warmup and not opts.consecutive_line_sampler: dataloader.sampler.set_epoch(epoch) # necessary for shuffling
    
        if track_progress and self.local_rank == 0:
            tr_progress_bar = tqdm(range(0, len(dataloader)))
        
        cnt = 0
        for i, batch in enumerate(dataloader):

            if opts.small_training:
                if i==opts.num_iters_small_training*opts.accu_steps:
                    break
            
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('load time:%.2f'%(time.time()-start_time))

            if not warmup:
                self.model.module.progress = float(self.model.total_steps) /\
                                               self.model.final_steps
                self.select_loss_indicator(i) # Alternate between loss_select = 0 and loss_select = 1 (condition i % 2 == 0)
                self.update_root_indicator(i) # Whether to update root pose
                self.update_body_indicator(i) # Whether to update body pose
                self.update_shape_indicator(i) # Whether to update shape
                self.update_cvf_indicator(i) # Whether to update canoical volume features

#                rtk_all = self.model.module.compute_rts()
#                self.model.module.rtk_all = rtk_all.clone()
#
#            # change near-far plane for all views
#            if self.model.module.progress>=opts.nf_reset:
#                rtk_all = rtk_all.detach().cpu().numpy()
#                valid_rts = self.model.module.latest_vars['idk'].astype(bool)
#                self.model.module.latest_vars['rtk'][valid_rts,:3] = rtk_all[valid_rts]
#                self.model.module.near_far.data = get_near_far(
#                                              self.model.module.near_far.data,
#                                              self.model.module.latest_vars)
#
#            self.optimizer.zero_grad()
            #try:
            total_loss,aux_out = self.model(batch)
            total_loss = total_loss/self.accu_steps

            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('forward time:%.2f'%(time.time()-start_time))
            #except:
            #    print("Failed forward pass")
            #    torch.cuda.synchronize()

            #try:
            total_loss.mean().backward()

            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

            if (i+1)%self.accu_steps == 0:
                self.clip_grad(aux_out)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if aux_out['nerf_root_rts_g']>1*opts.clip_scale and \
                            self.model.total_steps>opts.num_iters_small_training*self.accu_steps:
                    latest_path = '%s/params_latest.pth'%(self.save_dir)
                    self.load_network(latest_path, is_eval=False, rm_prefix=False)
            
            

            for i,param_group in enumerate(self.optimizer.param_groups):
                aux_out['lr_%02d'%i] = param_group['lr']

            if opts.nerfies_window: # c2f strategy # TODO: How to set num epochs correctly?
                n_total_alpha = 2 * 120 * min(opts.num_iters_small_training,len(self.trainloader)) * opts.accu_steps
                min_alpha = 8
                alpha = max(min_alpha, min(min_alpha+(self.model.module.num_freqs-min_alpha)*\
                            float(self.model.module.total_steps)/float(n_total_alpha), self.model.module.num_freqs))
                self.model.module.embedding_xyz.alpha = alpha
                self.model.module.embedding_dir.alpha = alpha

            self.model.module.total_steps += 1
            self.model.module.counter_frz_rebone -= 1./self.model.final_steps
            aux_out['counter_frz_rebone'] = self.model.module.counter_frz_rebone
            aux_out['alpha'] = self.model.module.embedding_xyz.alpha.item()

            if opts.local_rank==0: 
                self.save_logs(log, aux_out, self.model.module.total_steps, epoch)
            
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('total step time:%.2f'%(time.time()-start_time))
                torch.cuda.synchronize()
                start_time = time.time()

            cnt += 1
            if track_progress and self.local_rank == 0:
                tr_progress_bar.update(1)
                tr_progress_bar.set_description(
                    f"Training epoch {epoch} [{cnt}/{len(dataloader)}]"
                )
                tr_progress_bar.refresh()
            
    def train_one_epoch_rec(self, epoch, log, warmup=False, track_progress=False):
        opts = self.opts
        self.model.train()
        dataloader = self.trainloader
    
        if track_progress and self.local_rank == 0:
            tr_progress_bar = tqdm(range(0, len(dataloader)))
        
        cnt = 0
        for i, batch in enumerate(dataloader):

#           self.optimizer.zero_grad()
            total_loss,aux_out = [],[]
            batch_t = []
            for j in range(len(batch)):
                batch_t.append(batch[j].to(self.device))
            
            total_loss, aux_out = self.model(batch, self.model.module.total_steps)

            #with profile(activities=[
            #        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            #    with record_function("model_inference"):
            #        total_loss,aux_out = self.model(batch, self.model.module.total_steps)
            #
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            total_loss = total_loss/self.accu_steps
            total_loss.mean().backward()
            
            if (i+1)%self.accu_steps == 0:
                self.clip_grad(aux_out)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if aux_out['nerf_root_rts_g']>1*opts.clip_scale and \
                                self.model.total_steps>opts.num_iters_small_training*self.accu_steps:
                    latest_path = '%s/params_latest.pth'%(self.save_dir)
                    self.load_network(latest_path, is_eval=False, rm_prefix=False)
                
            for i,param_group in enumerate(self.optimizer.param_groups):
                aux_out['lr_%02d'%i] = param_group['lr']

            self.model.module.total_steps += 1
            self.model.module.counter_frz_rebone -= 1./self.model.final_steps
            aux_out['counter_frz_rebone'] = self.model.module.counter_frz_rebone

            if opts.local_rank==0: 
                self.save_logs(log, aux_out, self.model.module.total_steps, 
                        epoch)
            
            cnt += 1
            if track_progress and self.local_rank == 0:
                tr_progress_bar.set_description(
                    f"Training epoch {epoch} [{cnt}/{len(dataloader)}]"
                )
                tr_progress_bar.refresh()

    def update_cvf_indicator(self, i):
        """
        whether to update canoical volume features
        0: update all
        1: freeze 
        """
        opts = self.opts

        # during kp reprojection optimization
        if (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress < (opts.proj_start+opts.proj_end)):
            self.model.module.cvf_update = 1
        else:
            self.model.module.cvf_update = 0
        
        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.cvf_update = 1

        if opts.freeze_cvf:
            self.model.module.cvf_update = 1
    
    def update_shape_indicator(self, i):
        """
        whether to update shape
        0: update all
        1: freeze shape
        """
        opts = self.opts
        # incremental optimization
        # or during kp reprojection optimization
        if (opts.model_path!='' and \
        self.model.module.progress < opts.warmup_steps)\
         or (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress <(opts.proj_start + opts.proj_end)):
            self.model.module.shape_update = 1
        else:
            self.model.module.shape_update = 0

        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.shape_update = 1

        if opts.freeze_shape:
            self.model.module.shape_update = 1
    
    def update_root_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if (opts.freeze_proj and \
            opts.root_stab and \
           self.model.module.progress >=(opts.frzroot_start) and \
           self.model.module.progress <=(opts.proj_start + opts.proj_end+0.01))\
           : # to stablize
            self.model.module.root_update = 0
        else:
            self.model.module.root_update = 1
        
        # freeze shape after rebone        
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.root_update = 0
        
        if opts.freeze_root: # to stablize
            self.model.module.root_update = 0
    
    def update_body_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if opts.freeze_proj and \
           self.model.module.progress <=opts.frzbody_end: 
            self.model.module.body_update = 0
        else:
            self.model.module.body_update = 1

        
    def select_loss_indicator(self, i):
        """
        0: flo
        1: flo/sil/rgb
        """
        opts = self.opts
        if not opts.root_opt or \
            self.model.module.progress > (opts.warmup_steps):
            self.model.module.loss_select = 1
        elif i%2 == 0:
            self.model.module.loss_select = 0
        else:
            self.model.module.loss_select = 1

        #self.model.module.loss_select=1
        

    def reset_hparams(self, epoch):
        """
        reset hyper-parameters based on current geometry / cameras
        """
        opts = self.opts
        mesh_rest = self.model.latest_vars['mesh_rest']

        # reset object bound, for feature matching
        if epoch>int(self.num_epochs*(opts.bound_reset)):
            if mesh_rest.vertices.shape[0]>100:
                self.model.latest_vars['obj_bound'] = 1.2*np.abs(mesh_rest.vertices).max(0)
        
        # reinit bones based on extracted surface
        # only reinit for the initialization phase
        if opts.lbs and opts.model_path=='' and \
                        (epoch==int(self.num_epochs*opts.reinit_bone_steps) or\
                         epoch==0 or\
                         epoch==int(self.num_epochs*opts.warmup_steps)//2) and \
                        (opts.dfm_type != "quadratic" or opts.warmup_steps > 0.01):
            reinit_bones(self.model.module, mesh_rest, opts.num_bones)
            self.init_training() # add new params to optimizer
            if epoch>0:
                # freeze weights of root pose in the following 1% iters
                self.model.module.counter_frz_rebone = 0.01
                #reset error stats
                self.model.module.latest_vars['fp_err']      [:]=0
                self.model.module.latest_vars['flo_err']     [:]=0
                self.model.module.latest_vars['sil_err']     [:]=0
                self.model.module.latest_vars['flo_err_hist'][:]=0

        # need to add bones back at 2nd opt
        if opts.model_path!='':
            self.model.module.nerf_models['bones'] = self.model.module.bones

        # add nerf-skin when the shape is good
        if opts.lbs and opts.nerf_skin and \
                epoch==int(self.num_epochs*opts.dskin_steps):
            self.model.module.nerf_models['nerf_skin'] = self.model.module.nerf_skin # What is nerf skin?

        if not opts.finer_reconstruction:
            self.broadcast()

    def broadcast(self):
        """
        broadcast variables to other models
        """
        dist.barrier()
        if self.opts.lbs:
            dist.broadcast_object_list(
                    [self.model.module.num_bones, 
                    self.model.module.num_bone_used,],
                    0)
            dist.broadcast(self.model.module.bones,0)
            dist.broadcast(self.model.module.nerf_body_rts[1].rgb[0].weight, 0)
            dist.broadcast(self.model.module.nerf_body_rts[1].rgb[0].bias, 0)

        dist.broadcast(self.model.module.near_far,0)
   
    def clip_grad(self, aux_out):
        """
        gradient clipping
        """
        is_invalid_grad=False
        grad_nerf_coarse=[]
        grad_nerf_beta=[]
        grad_nerf_feat=[]
        grad_nerf_beta_feat=[]
        grad_nerf_fine=[]
        grad_nerf_unc=[]
        grad_nerf_flowbw=[]
        grad_nerf_skin=[]
        grad_nerf_vis=[]
        grad_nerf_root_rts=[]
        grad_nerf_body_rts=[]
        grad_root_code=[]
        grad_pose_code=[]
        grad_env_code=[]
        grad_vid_code=[]
        grad_bones=[]
        grad_skin_aux=[]
        grad_ks=[]
        grad_nerf_dp=[]
        grad_csenet=[]
        grad_mesh_nrd=[]
        
        for name,p in self.model.named_parameters():
            try: 
                pgrad_nan = p.grad.isnan()
                if pgrad_nan.sum()>0: 
                    # print(name)
                    is_invalid_grad=True
            except: pass
            if 'nerf_coarse' in name and 'beta' not in name:
                grad_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                grad_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name:
                grad_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                grad_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                grad_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                grad_nerf_unc.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                grad_nerf_flowbw.append(p)
            elif 'nerf_skin' in name:
                grad_nerf_skin.append(p)
            elif 'nerf_vis' in name:
                grad_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                grad_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                grad_nerf_body_rts.append(p)
            elif 'root_code' in name:
                grad_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                grad_pose_code.append(p)
            elif 'env_code' in name:
                grad_env_code.append(p)
            elif 'vid_code' in name:
                grad_vid_code.append(p)
            elif 'module.bones' == name:
                grad_bones.append(p)
            elif 'module.skin_aux' == name:
                grad_skin_aux.append(p)
            elif 'module.ks_param' == name:
                grad_ks.append(p)
            elif 'nerf_dp' in name:
                grad_nerf_dp.append(p)
            elif 'csenet' in name:
                grad_csenet.append(p)
            else: continue

        if "nerf_dfm_mesh_opt" in self.model.nerf_models.keys():
            modules_nr_extra_dfm = [self.model.nerf_dfm_mesh_opt]
            for submodule in modules_nr_extra_dfm:
                for name, p in submodule.named_parameters():    
                    grad_mesh_nrd.append(p)
        
        # freeze root pose when using re-projection loss only
        if self.model.module.root_update == 0:
            self.zero_grad_list(grad_root_code)
            self.zero_grad_list(grad_nerf_root_rts)
        if self.model.module.body_update == 0:
            self.zero_grad_list(grad_pose_code)
            self.zero_grad_list(grad_nerf_body_rts)
        if self.opts.freeze_body_mlp:
            self.zero_grad_list(grad_nerf_body_rts)
        if self.model.module.shape_update == 1:
            self.zero_grad_list(grad_nerf_coarse)
            self.zero_grad_list(grad_nerf_beta)
            self.zero_grad_list(grad_nerf_vis)
            #TODO add skinning 
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_nerf_skin)
            self.zero_grad_list(grad_skin_aux)
        if self.model.module.cvf_update == 1:
            self.zero_grad_list(grad_nerf_feat)
            self.zero_grad_list(grad_nerf_beta_feat)
            self.zero_grad_list(grad_csenet)
        if self.opts.freeze_coarse:
            # freeze shape
            # this include nerf_coarse, nerf_skin (optional)
            grad_coarse_mlp = []
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_coarse)
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_skin)
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_feat)
            self.zero_grad_list(grad_coarse_mlp)

            #self.zero_grad_list(grad_nerf_coarse) # freeze shape

            # freeze skinning 
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_skin_aux)
            #self.zero_grad_list(grad_nerf_skin) # freeze fine shape

            ## freeze pose mlp
            #self.zero_grad_list(grad_nerf_body_rts)

            # add vis
            self.zero_grad_list(grad_nerf_vis)
            #print(self.model.module.nerf_coarse.xyz_encoding_1[0].weight[0,:])
        
        clip_scale=self.opts.clip_scale
 
        #TODO don't clip root pose
        aux_out['nerf_coarse_g']   = clip_grad_norm_(grad_nerf_coarse,    1*clip_scale).item()
        aux_out['nerf_beta_g']     = clip_grad_norm_(grad_nerf_beta,      1*clip_scale).item()
        aux_out['nerf_feat_g']     = clip_grad_norm_(grad_nerf_feat,     .1*clip_scale).item()
        aux_out['nerf_beta_feat_g']= clip_grad_norm_(grad_nerf_beta_feat,.1*clip_scale).item()
        aux_out['nerf_fine_g']     = clip_grad_norm_(grad_nerf_fine,     .1*clip_scale).item()
        aux_out['nerf_unc_g']     = clip_grad_norm_(grad_nerf_unc,       .1*clip_scale).item()
        aux_out['nerf_flowbw_g']   = clip_grad_norm_(grad_nerf_flowbw,   .1*clip_scale).item()
        aux_out['nerf_skin_g']     = clip_grad_norm_(grad_nerf_skin,     .1*clip_scale).item()
        aux_out['nerf_vis_g']      = clip_grad_norm_(grad_nerf_vis,      .1*clip_scale).item()
        aux_out['nerf_root_rts_g'] = clip_grad_norm_(grad_nerf_root_rts,100*clip_scale).item()
        aux_out['nerf_body_rts_g'] = clip_grad_norm_(grad_nerf_body_rts,100*clip_scale).item()
        aux_out['root_code_g']= clip_grad_norm_(grad_root_code,          .1*clip_scale).item()
        aux_out['pose_code_g']= clip_grad_norm_(grad_pose_code,         100*clip_scale).item()
        aux_out['env_code_g']      = clip_grad_norm_(grad_env_code,      .1*clip_scale).item()
        aux_out['vid_code_g']      = clip_grad_norm_(grad_vid_code,      .1*clip_scale).item()
        aux_out['bones_g']         = clip_grad_norm_(grad_bones,          1*clip_scale).item()
        aux_out['skin_aux_g']   = clip_grad_norm_(grad_skin_aux,         .1*clip_scale).item()
        aux_out['ks_g']            = clip_grad_norm_(grad_ks,            .1*clip_scale).item()
        aux_out['nerf_dp_g']       = clip_grad_norm_(grad_nerf_dp,       .1*clip_scale).item()
        aux_out['csenet_g']        = clip_grad_norm_(grad_csenet,        .1*clip_scale).item()
        aux_out["mesh_nrd_g"]      = clip_grad_norm_(grad_mesh_nrd,      .1*clip_scale).item()

        #if aux_out['nerf_root_rts_g']>10:
        #    is_invalid_grad = True
        if is_invalid_grad:
            print("Encountered invalid grad")
            self.zero_grad_list(self.model.parameters())
            
    @staticmethod
    def find_nerf_coarse(nerf_model):
        """
        zero grad for coarse component connected to inputs, 
        and return intermediate params
        """
        param_list = []
        input_layers=[0]+nerf_model.skips

        input_wt_names = []
        for layer in input_layers:
            input_wt_names.append(f"xyz_encoding_{layer+1}.0.weight")

        for name,p in nerf_model.named_parameters():
            if name in input_wt_names:
                # get the weights according to coarse posec
                # 63 = 3 + 60
                # 60 = (num_freqs, 2, 3)
                out_dim = p.shape[0]
                pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
                # TODO
                num_coarse = 8 # out of 10
                #num_coarse = 10 # out of 10
                #num_coarse = 1 # out of 10
           #     p.grad[:,:3] = 0 # xyz
           #     p.grad[:,3:pos_dim].view(out_dim,-1,6)[:,:num_coarse] = 0 # xyz-coarse
                p.grad[:,pos_dim:] = 0 # others
            else:
                param_list.append(p)
        return param_list

    @staticmethod 
    def render_vid(model, batch):
        opts=model.opts
        model.set_input(batch)
        rtk = model.rtk
        kaug=model.kaug.clone()
        embedid=model.embedid

        rendered, _ = model.nerf_render(rtk, kaug, embedid, ndepth=opts.ndepth)
        if 'xyz_camera_vis' in rendered.keys():    del rendered['xyz_camera_vis']   
        if 'xyz_canonical_vis' in rendered.keys(): del rendered['xyz_canonical_vis']
        if 'pts_exp_vis' in rendered.keys():       del rendered['pts_exp_vis']      
        if 'pts_pred_vis' in rendered.keys():      del rendered['pts_pred_vis']     
        rendered_first = {}
        for k,v in rendered.items():
            if v.dim()>0: 
                bs=v.shape[0]
                rendered_first[k] = v[:bs//2] # remove loss term
        return rendered_first 

    @staticmethod
    def extract_mesh(model,chunk,grid_size,
                      #threshold = -0.005,
                      threshold = -0.002,
                      #threshold = 0.,
                      embedid=None,
                      mesh_dict_in=None,
                      keep_original_mesh=False,
                      bound=None):
        opts = model.opts
        mesh_dict = {}
        if bound is None:
            if model.near_far is not None: 
                bound = model.latest_vars['obj_bound']
            else: 
                if not (opts.fine_nerf_net and opts.use_sdf_finenerf):
                    bound=1.5*np.asarray([1,1,1])
                else:
                    bound=4*np.asarray([1,1,1])

        mesh_dict_in = None
        if mesh_dict_in is None:
            ptx = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
            pty = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
            ptz = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
            query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
            query_yxz = torch.Tensor(query_yxz.reshape(-1,3))
            #pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
            #query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)

            bs_pts = query_yxz.shape[0]
            out_chunks = []
            for i in range(0, bs_pts, chunk):
                query_yxz_chunk = query_yxz[i:i+chunk]
                query_xyz_chunk = torch.cat([query_yxz_chunk[:,1:2], query_yxz_chunk[:,0:1], query_yxz_chunk[:,2:3]],-1)
                query_xyz_chunk = torch.Tensor(query_xyz_chunk).to(model.device).view(-1, 3)
                query_dir_chunk = torch.zeros_like(query_xyz_chunk)
                
                # backward warping 
                if embedid is not None and not opts.queryfw:
                    query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                                   query_xyz_chunk, embedid)
                if opts.symm_shape: 
                    #TODO set to x-symmetric
                    query_xyz_chunk[...,0] = query_xyz_chunk[...,0].abs()

                if opts.IPE:
                    xyz_embedded = integrated_pos_enc( (query_xyz_chunk.resize(query_xyz_chunk.size(0), 1, query_xyz_chunk.size(1)), torch.zeros((query_xyz_chunk.size(0), 1, query_xyz_chunk.size(1)), device="cuda:0")), 0, 16)
                    xyz_embedded = xyz_embedded.resize(xyz_embedded.size(0), xyz_embedded.size(2))
                else:
                    xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                
                if opts.use_sdf_finenerf and opts.fine_nerf_net:
                    out_chunks += [model.nerf_fine(xyz_embedded, sigma_only=True).cpu()]
                else:
                    out_chunks += [model.nerf_coarse(xyz_embedded, sigma_only=True).cpu()]
            vol_o = torch.cat(out_chunks, 0)
            vol_o = vol_o.view(grid_size, grid_size, grid_size)
            #vol_o = F.softplus(vol_o)

            if False: #not opts.full_mesh:
                #TODO set density of non-observable points to small value
                if model.latest_vars['idk'].sum()>0:
                    vis_chunks = []
                    for i in range(0, bs_pts, chunk):
                        query_yxz_chunk = query_yxz[i:i+chunk]
                        query_xyz_chunk = torch.cat([query_yxz_chunk[:,1:2], query_yxz_chunk[:,0:1], query_yxz_chunk[:,2:3]],-1)
                        query_xyz_chunk = torch.Tensor(query_xyz_chunk).to(model.device).view(-1, 3)

                        if opts.nerf_vis:
                            # this leave no room for halucination and is not what we want
                            if opts.IPE:
                                xyz_embedded = integrated_pos_enc( (query_xyz_chunk.resize(query_xyz_chunk.size(0), 1, query_xyz_chunk.size(1)), torch.zeros((query_xyz_chunk.size(0), 1, query_xyz_chunk.size(1)), device="cuda:0")), 0, 16)
                                xyz_embedded = xyz_embedded.resize(xyz_embedded.size(0), xyz_embedded.size(2))
                            else:
                                xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)

                            vis_chunk_nerf = model.nerf_vis(xyz_embedded)
                            vis_chunk = vis_chunk_nerf[...,0].sigmoid()
                        else:
                            #TODO deprecated!
                            vis_chunk = compute_point_visibility(query_xyz_chunk.cpu(),
                                             model.latest_vars, model.device)[None]
                        vis_chunks += [vis_chunk.cpu()]
                    vol_visi = torch.cat(vis_chunks, 0)
                    vol_visi = vol_visi.view(grid_size, grid_size, grid_size)
                    vol_o[vol_visi<0.5] = -1
            else:
                print("Not using vis for mesh")

            ## save color of sampled points 
            #cmap = cm.get_cmap('cool')
            ##pts_col = cmap(vol_visi.float().view(-1).cpu())
            #pts_col = cmap(vol_o.sigmoid().view(-1).cpu())
            #mesh = trimesh.Trimesh(query_xyz.view(-1,3).cpu(), vertex_colors=pts_col)
            #mesh.export('0.obj')
            #pdb.set_trace()

            # print('fraction occupied:', (vol_o > threshold).float().mean())
            
            vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy().astype(np.float32), threshold)
            # Don't scale -> we want it later
            vertices = (vertices - grid_size/2)/grid_size*2*bound[None, :]
            mesh = trimesh.Trimesh(vertices, triangles)

            # mesh post-processing 
            if len(mesh.vertices)>0:
                if opts.use_cc:
                    # keep the largest mesh
                    mesh = [i for i in mesh.split(only_watertight=False)]
                    mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                    mesh = mesh[-1]

                # assign color based on canonical location
                vis = mesh.vertices
                try:
                    model.module.vis_min = vis.min(0)[None]
                    model.module.vis_len = vis.max(0)[None] - vis.min(0)[None]
                except: # test time
                    model.vis_min = vis.min(0)[None]
                    model.vis_len = vis.max(0)[None] - vis.min(0)[None]
                vis = vis - model.vis_min
                vis = vis / model.vis_len
                if not opts.ce_color:
                    vis = get_vertex_colors(model, mesh, frame_idx=0)
                mesh.visual.vertex_colors[:,:3] = vis*255

                #if keep_original_mesh:
                #    mesh.vertices = (mesh.vertices * grid_size*2*bound[None, :]) + grid_size/2

        # forward warping: TODO: Check for correct forward warping
        #if embedid is not None and opts.queryfw:
        #    mesh = mesh_dict_in['mesh'].copy()
        #    vertices = mesh.vertices
        #    vertices, mesh_dict = warp_fw(opts, model, mesh_dict, 
        #                                   vertices, embedid)
        #    mesh.vertices = vertices
        mesh_dict['mesh'] = mesh
        return mesh_dict

    def debug_view_synthesis_progress(self, epoch, absolute_epoch, w_extra_dfm=True, frames_to_render=20, extra_suffix=""):
        
        #if len(self.model.latest_vars["mesh_rest"].vertices) < 100:
        #    print("Could not synthesize views as could not compute near/far")
        #    return

        with torch.no_grad():
            eval_opts = deepcopy(flags.FLAGS)
            # TODO: Add other flags when debugging synthesis
            save_dir = os.path.join("logdir", self.opts.logname)
            save_dir_imgs = os.path.join("figures", self.opts.logname)
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_dir_imgs, exist_ok=True)

            eval_opts.extra_dfm_nr = w_extra_dfm
            eval_opts.chunk_mult_eval_mlp = 1

            model = self.model
            model.eval()
            # eval_opts.rnd_frame_chunk = 2
            # trainer.eval()

            # TODO: Check how many frame ids / vid ids are there and select random ones and fixed ones?
            vidid = 0
            image_scale = 0.5

            # select video to render
            vidid = vidid # 0 to 10

            # select rendering resolution
            raw_res = (1080*image_scale,1920*image_scale) # h, w
            target_size = (int(raw_res[0]), 
                        int(raw_res[1])) 

            # get frame id 
            if eval_opts.seqname == "cat-pikachiu":
                frames = [4, 83, 679]
                vidids = [0, 0, 7]
            else:
                frames = [3, 20, 5]
                #frames = [3, 206, 74]
                vidids = [1, 2, 3]
            
            for idx_frame_id in tqdm(range(len(frames))):
            
                frameid = torch.Tensor([frames[idx_frame_id]]).to(model.device).long()
                vidid = vidids[idx_frame_id]
                bs = 1 #len(frameid)

                # query root poses
                # bs, 4,4 (R|T) # extrinsics
                #         (f|p) # intrinsics
                with torch.no_grad():
                    rtks = torch.eye(4)[None].repeat(bs,1,1)
                    root_rts = model.nerf_root_rts(frameid)
                    rtk_base = model.create_base_se3(bs, model.device)
                    rtks[:,:3] = model.refine_rt(rtk_base, root_rts)
                    rtks[:,3,:] = model.ks_param[vidid]
                    rtks[:,3] = rtks[:,3]*image_scale

                # compute near-far plane
                near_far = torch.zeros(bs,2).to(model.device)
                vars_np = {}
                vars_np['rtk'] = rtks.cpu().numpy()
                vars_np['idk'] = np.ones(bs)
                near_far = self.model.near_far
                
                """ get_near_far(near_far,
                                        vars_np,
                                        pts=self.model.latest_vars["mesh_rest"].vertices) """

                # render
                render_frame_id = frameid #[i:i+1]
                # print('rendering video %02d, frame %03d/%03d'%(vidid, i,len(frameid)))

                rndmask = np.ones((target_size[0], target_size[1]))
                rays = construct_rays_nvs(target_size, rtks, 
                                                near_far[frames[idx_frame_id]:frames[idx_frame_id]+1], rndmask, model.device)
                # query env code
                rays['env_code'] = model.env_code(render_frame_id)[:,None]
                rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

                # query deformation
                time_embedded = model.pose_code(render_frame_id)[:,None]
                rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)

                if eval_opts.use_separate_code_dfm:
                    if eval_opts.combine_dfm_and_pose:
                        dfm_embedded = model.dfm_code_w_pose(render_frame_id)[:, None]
                    else:
                        dfm_embedded = model.dfm_code(render_frame_id)[:, None]
                    rays['dfm_embedded'] = dfm_embedded.repeat(1,rays['nsample'],1)

                bone_rts = model.nerf_body_rts(render_frame_id)
                rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
                nerf_models = model.nerf_models
                embeddings = model.embeddings # Positional embeddings

                # render images only
                results=defaultdict(list)
                bs_rays = rays['bs'] * rays['nsample'] #
                chunk_size = self.opts.chunk//self.opts.div_factor_eval_chunk
                for j in range(0, bs_rays, chunk_size):
                    rays_chunk = chunk_rays(rays, j, chunk_size)
                    model.update_delta_rts(rays_chunk)
                    rendered_chunks = render_rays(nerf_models,
                                embeddings,
                                rays_chunk,
                                N_samples = self.opts.ndepth,
                                perturb=0,
                                noise_std=0,
                                chunk=chunk_size, # chunk size is effective in val mode
                                use_fine=True,
                                img_size=model.img_size,
                                obj_bound = model.latest_vars['obj_bound'],
                                render_vis=True,
                                opts=eval_opts,
                                render_losses=False
                                )
                    for k, v in rendered_chunks.items():
                        results[k] += [v.cpu()]

                for k, v in results.items():
                    v = torch.cat(v, 0)
                    v = v.view(rays['nsample'], -1)
                    results[k] = v

                #original_rgb = results['img_at_samp'].numpy().reshape(target_size[0], target_size[1],3)
                #original_sil = results['sil_at_samp'][...,0].numpy().reshape(target_size[0], target_size[1])
                rgb = results['img_coarse'].numpy().reshape(target_size[0], target_size[1],3)
                sil = results['sil_coarse'][...,0].numpy().reshape(target_size[0], target_size[1])
                rgb[sil<0.5] = 0
                cv2.imwrite(os.path.join(save_dir_imgs, f"synth_rgb_{absolute_epoch}-{epoch}_vid{vidid}_f{render_frame_id.item()}_{extra_suffix}.jpg"), (rgb[:,:,::-1]*255).astype(np.uint8) )
                cv2.imwrite(os.path.join(save_dir_imgs, f"synth_sil_{absolute_epoch}-{epoch}_vid{vidid}_f{render_frame_id.item()}_{extra_suffix}.jpg"), (sil*255).astype(np.uint8))
                #cv2.imwrite(os.path.join(save_dir_imgs, f"org_rgb_{absolute_epoch}-f{render_frame_id.item()}.jpg"), original_rgb[:,:,::-1]*255)
                #cv2.imwrite(os.path.join(save_dir_imgs, f"org_sil_{absolute_epoch}-f{render_frame_id.item()}.jpg"), original_sil*255)

                if "img_fine" in results.keys():
                    rgb_fine = results['img_fine'].numpy().reshape(target_size[0], target_size[1],3)
                    rgb_fine[sil<0.5] = 0
                    cv2.imwrite(os.path.join(save_dir_imgs, f"synth_rgb_fine_{absolute_epoch}-{epoch}_vid{vidid}_f{render_frame_id.item()}_{extra_suffix}.jpg"), (rgb_fine[:,:,::-1]*255).astype(np.uint8) )

            model.train()

    def debug_vector_field(self, epoch, absolute_epoch):
        
        #if len(self.model.latest_vars["mesh_rest"].vertices) < 100:
        #    print("Could not synthesize views as could not compute near/far")
        #    return

        with torch.no_grad():
            eval_opts = deepcopy(flags.FLAGS)
            # TODO: Add other flags when debugging synthesis
            save_dir = os.path.join("logdir", self.opts.logname)
            save_dir_imgs = os.path.join("figures", self.opts.logname)
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_dir_imgs, exist_ok=True)

            eval_opts.extra_dfm_nr = True
            eval_opts.chunk_mult_eval_mlp = 1

            model = self.model
            model.eval()
            # eval_opts.rnd_frame_chunk = 2
            # trainer.eval()

            # TODO: Check how many frame ids / vid ids are there and select random ones and fixed ones?
            vidid = 0
            image_scale = 0.5
            with torch.no_grad():
                mesh = self.extract_mesh(self.model, eval_opts.chunk, eval_opts.sample_grid3d, eval_opts.mc_threshold, keep_original_mesh=True)

                grid_size = 16
                embedid = 1

                bound = np.array(self.model.latest_vars["obj_bound"])
                limit = np.max(np.abs(bound))*2

                enc_temp = model.pose_code(torch.tensor([embedid], device=model.device))
                ptx = np.linspace(-limit, limit, grid_size).astype(np.float32)
                pty = np.linspace(-limit, limit, grid_size).astype(np.float32)
                ptz = np.linspace(-limit, limit, grid_size).astype(np.float32)
                query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
                #pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
                #query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
                query_xyz = torch.Tensor(query_yxz).view(-1, 3)
                #query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

                bs_pts = query_xyz.shape[0]
                out_chunks = []
                chunk_size = 32
                for i in range(0, bs_pts, chunk_size):
                    query_xyz_chunk = query_xyz[i:i+chunk_size].to(model.device)

                    """ # backward warping 
                    if embedid is not None and not opts.queryfw:
                        query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                                        query_xyz_chunk, embedid) """
                    
                    enc_spat = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                    enc_temp_f = enc_temp.repeat(chunk_size,1)
                    mesh_def_encoding = torch.cat([enc_spat, enc_temp_f], 1)
                    pts_dfm = self.model.nerf_models["extra_deform_net"](mesh_def_encoding)
                    # Here if quadratic convert from parameters to point deformation
                    if eval_opts.dfm_type == "quadratic":
                        quad_params = pts_dfm.resize(pts_dfm.size(0), 3, 9)
                        coords_quad = query_xyz_chunk.unsqueeze(1)
                        coords_quad = torch.cat([coords_quad, coords_quad**2, 
                                        (coords_quad[:,:,0]*coords_quad[:,:,1]).unsqueeze(-1), 
                                        (coords_quad[:,:,1]*coords_quad[:,:,2]).unsqueeze(-1), 
                                        (coords_quad[:,:,0]*coords_quad[:,:,2]).unsqueeze(-1)], 2)
                        coords_quad = coords_quad.repeat(1, 3, 1)
                        pts_dfm = (quad_params*coords_quad).sum(2)

                    out_chunks += [pts_dfm.cpu()]

                vol_o = torch.cat(out_chunks, 0)
                vol_o = vol_o.view(grid_size, grid_size, grid_size, 3)

            query_xyz_np = query_xyz.cpu().numpy()
            vol_o_np = vol_o.cpu().numpy()

            # Debug figure 1
            fig = go.Figure(data = [
                go.Mesh3d(
                    x=np.array(mesh["mesh"].vertices[:, 0]), 
                    y=np.array(mesh["mesh"].vertices[:, 1]), 
                    z=np.array(mesh["mesh"].vertices[:, 2]), 
                    color='green', opacity=0.10),
                go.Cone(
                    y=query_xyz_np[:, 0],
                    z=query_xyz_np[:, 1],
                    x=query_xyz_np[:, 2],
                    u=vol_o_np[:,:,:,0].flatten(),
                    v=vol_o_np[:,:,:,1].flatten(),
                    w=vol_o_np[:,:,:,2].flatten(),
                    colorscale='Blues',
                    sizemode="absolute",
                    sizeref=0.04)]
            )

            fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                        camera_eye=dict(x=-0.5, y=0.1, z=0.2)))

            save_name = os.path.join(save_dir_imgs, f"vf_{absolute_epoch}-{epoch}_vid{vidid}_1.jpg")
            fig.write_image(save_name)

            # Debug figure 2
            fig = go.Figure(data = [
                go.Mesh3d(
                    x=np.array(mesh["mesh"].vertices[:, 0]), 
                    y=np.array(mesh["mesh"].vertices[:, 1]), 
                    z=np.array(mesh["mesh"].vertices[:, 2]), 
                    color='green', opacity=0.10),
                go.Cone(
                    y=query_xyz_np[:, 0],
                    z=query_xyz_np[:, 1],
                    x=query_xyz_np[:, 2],
                    u=vol_o_np[:,:,:,0].flatten(),
                    v=vol_o_np[:,:,:,1].flatten(),
                    w=vol_o_np[:,:,:,2].flatten(),
                    colorscale='Blues',
                    sizemode="absolute",
                    sizeref=0.04)]
            )

            fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                        camera_eye=dict(x=0.35, y=0.1, z=0.2)))

            save_name = os.path.join(save_dir_imgs, f"vf_{absolute_epoch}-{epoch}_vid{vidid}_2.jpg")
            fig.write_image(save_name)

            model.train()

    def save_logs(self, log, aux_output, total_steps, epoch):
        for k,v in aux_output.items():
            self.add_scalar(log, k, aux_output,total_steps)
        
    def add_image_grid(self, rendered_seq, log, epoch):
        for k,v in rendered_seq.items():
            grid_img = image_grid(rendered_seq[k],3,3)
            if k=='depth_rnd':scale=True
            elif k=='occ':scale=True
            elif k=='unc_pred':scale=True
            elif k=='proj_err':scale=True
            elif k=='feat_err':scale=True
            else: scale=False
            self.add_image(log, k, grid_img, epoch, scale=scale)

    def add_image(self, log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """

        if self.isflow(tag):
            timg = timg.detach().cpu().numpy()
            timg = flow_to_image(timg)
        elif scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
        else:
            timg = torch.clamp(timg, 0,1)
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
            print('error'); pdb.set_trace()
        else:
            formats='HWC'

        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)

    @staticmethod
    def del_key(states, key):
        if key in states.keys():
            del states[key]
    
    @staticmethod
    def isflow(tag):
        flolist = ['flo_coarse', 'fdp_coarse', 'flo', 'fdp', 'flo_at_samp']
        if tag in flolist:
           return True
        else:
            return False

    @staticmethod
    def zero_grad_list(paramlist):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        for p in paramlist:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

