import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)

import torch
# import libs
import numpy as np
import cv2
import torch
from absl import flags
from collections import defaultdict

from nnutils import banmo
from utils.io import extract_data_info
from dataloader import frameloader
from nnutils.geom_utils import get_near_far, sample_xy, K2inv, raycast, chunk_rays
from nnutils.train_utils import v2s_trainer
from nnutils.rendering import render_rays
from copy import deepcopy
from tqdm import tqdm

opts = flags.FLAGS

# script specific ones
flags.DEFINE_string('path_new_ds', f'database/{opts.seqname}_synthetic_ds_nerf', 'path to new synthetic dataset')

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
  
# load flags
args = opts.read_flags_from_files(['--seqname=cat-pikachiu', 
                                   '--nouse_corresp', 
                                   '--nouse_unc', 
                                   '--perturb=0', 
                                   '--chunk=256'])
unknown_flags, unparsed_args = opts._parse_args(args, known_only=True)
opts.mark_as_parsed()
opts.validate_all_flags()

os.makedirs(opts.path_new_ds, exist_ok=True)

# load model
trainer = v2s_trainer(opts, is_eval=True)
data_info = trainer.init_dataset()
trainer.define_model(data_info)

model = trainer.model
model.eval()

nerf_models = model.nerf_models
embeddings = model.embedding

with torch.no_grad():
    eval_opts = deepcopy(opts)
    save_dir = os.path.join("logdir", opts.logname)
    args = eval_opts.read_flags_from_files([ 
                            f'--model_path={os.path.join(save_dir, "params_latest.pth")}', 
                            f'--seqname={eval_opts.seqname}',
                            '--nouse_corresp', 
                            '--nouse_unc', 
                            '--perturb=0', 
                            '--chunk=1024'])
    unknown_flags, unparsed_args = eval_opts._parse_args(args, known_only=True)
    eval_opts.mark_as_parsed()
    eval_opts.lineload = False
    eval_opts.validate_all_flags()

    # load model
    trainer = v2s_trainer(eval_opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    model = trainer.model
    model.eval()
    # TODO: Check how many frame ids / vid ids are there and select random ones and fixed ones?
    vidid = 0
    vid_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13]
    image_scale = 0.5

    # select video to render
    vidid = vidid # 0 to 10

    # select rendering resolution
    raw_res = (1080*image_scale,1920*image_scale) # h, w
    target_size = (int(raw_res[0]), 
                int(raw_res[1])) 

    # get frame id 
    start_idx = model.data_offset[vidid]
    end_idx = model.data_offset[vidid+1]
    sample_idx = np.asarray(range(start_idx,end_idx))
    frameid = torch.Tensor(sample_idx).to(model.device).long()
    bs = len(frameid)

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
    near_far = get_near_far(near_far,
                            vars_np,
                            pts=model.latest_vars['mesh_rest'].vertices)

    # render
    for n_seq in vid_ids:
        for i in tqdm(range(0, frameid)): # Generate different cameras
            render_frame_id = frameid[i:i+1]
            # print('rendering video %02d, frame %03d/%03d'%(vidid, i,len(frameid)))

            rndmask = np.ones((target_size[0], target_size[1]))
            rays = construct_rays_nvs(target_size, rtks[i:i+1], 
                                            near_far[i:i+1], rndmask, model.device)
            # query env code
            rays['env_code'] = model.env_code(render_frame_id)[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

            # query deformation
            time_embedded = model.pose_code(render_frame_id)[:,None]
            rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
            bone_rts = model.nerf_body_rts(render_frame_id)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
            model.update_delta_rts(rays)
            nerf_models = model.nerf_models
            embeddings = model.embeddings # Positional embeddings

            # render images only
            results=defaultdict(list)
            bs_rays = rays['bs'] * rays['nsample'] #
            for j in range(0, bs_rays, eval_opts.chunk):
                rays_chunk = chunk_rays(rays,j,eval_opts.chunk)
                rendered_chunks = render_rays(nerf_models,
                            embeddings,
                            rays_chunk,
                            N_samples = eval_opts.ndepth,
                            perturb=0,
                            noise_std=0,
                            chunk=eval_opts.chunk, # chunk size is effective in val mode
                            use_fine=True,
                            img_size=model.img_size,
                            obj_bound = model.latest_vars['obj_bound'],
                            render_vis=True,
                            opts=eval_opts,
                            )
                for k, v in rendered_chunks.items():
                    results[k] += [v.cpu()]

            for k, v in results.items():
                v = torch.cat(v, 0)
                v = v.view(rays['nsample'], -1)
                results[k] = v

            rgb = results['img_coarse'].numpy().reshape(target_size[0], target_size[1],3)
            sil = results['sil_coarse'][...,0].numpy().reshape(target_size[0], target_size[1])
            rgb[sil<0.5] = 0    
            #cv2.imwrite(os.path.join(save_dir, f"synth_rgb_{absolute_epoch}-{epoch}_vid{vidid}_f{render_frame_id.item()}.jpg"), (rgb[:,:,::-1]*255).astype(np.uint8) )
            #cv2.imwrite(os.path.join(save_dir, f"synth_sil_{absolute_epoch}-{epoch}_vid{vidid}_f{render_frame_id.item()}.jpg"), (sil*255).astype(np.uint8))
