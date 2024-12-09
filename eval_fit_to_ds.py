import os
# os.chdir('/home/aagudo/sergio_montoya/TFM')

# import libs
import numpy as np
import cv2
import torch
from absl import flags
from collections import defaultdict
# import mediapy

from nnutils import banmo
from utils.io import extract_data_info
from dataloader import frameloader
from nnutils.geom_utils import get_near_far, sample_xy, K2inv, raycast, chunk_rays
from nnutils.train_utils import v2s_trainer
from nnutils.rendering import render_rays
from nnutils.loss_utils import gkern
from absl import flags, app
import sys
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim

opts = flags.FLAGS
#flags.DEFINE_integer('n_render', 20, 'number of images to render')
#flags.DEFINE_integer('chunk_size', 256, 'chunk size of rays to render')
flags.DEFINE_string('flagfile_path', '', 'path to opts.log')
flags.DEFINE_string('model_path_load', '', 'path to params_latest.pth')
flags.DEFINE_string('seqname_str', 'cat-pikachiu', 'string of sequence name')
flags.DEFINE_float('img_scale_eval', 0.25, 'scale of image to render')

# script specific ones
def construct_rays_nvs(target_size, rtks, near_far, rndmask, device):
    """
    rndmask: controls which pixel to render
    """
    bs = rtks.shape[0]
    rtks = torch.Tensor(rtks).to(device)
    rndmask = torch.Tensor(rndmask).to(device).view(-1)>0

    img_size = max(target_size)
    _, xys = sample_xy(img_size, bs, 0, device, return_all=True)
    xys=xys.view(img_size,img_size,2)[:target_size[0], :target_size[1]].reshape(1,-1,2)
    xys = xys[:,rndmask]
    Rmat = rtks[:,:3,:3]
    Tmat = rtks[:,:3,3]
    Kinv = K2inv(rtks[:,3])
    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    return rays

def main(_):
  # load model
  exp_name = opts.model_path_load.split("/")[1]

  args = opts.read_flags_from_files([
                                    f'--flagfile={opts.flagfile_path}',
                                    f'--model_path={opts.model_path_load}',
                                    f'--seqname={opts.seqname_str}', 
                                   '--nouse_corresp', 
                                   '--nouse_unc', 
                                   '--perturb=0', 
                                   '--chunk=4096',
                                    '--finer_reconstruction=False',
                                    '--fr_ckpt=False'
                                  ])

  unknown_flags, unparsed_args = opts._parse_args(args, known_only=True)
  opts.mark_as_parsed()
  opts.validate_all_flags()

  trainer = v2s_trainer(opts, is_eval=True)
  data_info = trainer.init_dataset()    
  trainer.define_model(data_info)

  model = trainer.model
  model.eval()

  nerf_models = model.nerf_models
  embeddings = model.embeddings # positional encodings

  import matplotlib.pyplot as plt

  PSNRs_coarse = []
  img_ssim_mean_coarse = []
  hf_amp_mean_mse_coarse = []

  PSNRs = []
  hf_amp_mean_mse = []
  sil_mean_mse = []
  img_mean_mse = []
  img_ssim_mean = []
  img_scale = opts.img_scale_eval

  w = int(1920*img_scale)
  h = int(1080*img_scale)
  for i in range(len(trainer.evalloader.dataset.datasets)):
    trainer.evalloader.dataset.datasets[i].w = w
    trainer.evalloader.dataset.datasets[i].h = h
    trainer.evalloader.dataset.datasets[i].dont_crop = True

  path_name = f"img_syn/{exp_name}"
  os.makedirs(path_name, exist_ok=True)

  if opts.seqname == "cat-pikachiu" or opts.seqname == "cat-pikachiu-hd":
    resolutions = [
        [1920, 1080],
        [1080, 1920],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
        [1920, 1080],
    ]

  for item_i in tqdm(range(len(trainer.evalloader.dataset))):
    elem = trainer.evalloader.dataset[item_i]

    # get frame id 
    vidid = elem["dataid"][0]
    frameid = torch.Tensor([elem["frameid"][0] + model.data_offset[vidid]]).to(model.device).long()

    w = int(resolutions[vidid][0]*img_scale)
    h = int(resolutions[vidid][1]*img_scale)
    for i in range(len(trainer.evalloader.dataset.datasets)):
        trainer.evalloader.dataset.datasets[i].w = w
        trainer.evalloader.dataset.datasets[i].h = h
        trainer.evalloader.dataset.datasets[i].dont_crop = True

    with torch.no_grad():
        bs = 1
        rtks = torch.eye(4)[None].repeat(bs,1,1)
        root_rts = model.nerf_root_rts(frameid)
        rtk_base = model.create_base_se3(bs, model.device)
        rtks[:,:3] = model.refine_rt(rtk_base, root_rts)
        rtks[:,3,:] = model.ks_param[vidid]
        rtks[:,3] = rtks[:,3]*img_scale

    # compute near-far plane
    near_far = torch.zeros(1,2).to(model.device)
    vars_np = {}
    vars_np['rtk'] = rtks.cpu().numpy()
    vars_np['idk'] = np.ones(1)
    #near_far = get_near_far(near_far,
    #                        vars_np,
    #                        pts=model.latest_vars['mesh_rest'].vertices)

    raw_res = (h, w) # h, w
    target_size = (int(raw_res[0]), 
                int(raw_res[1])) 

    near_far = model.near_far[frameid]
    rndmask = np.ones(target_size)
    rays = construct_rays_nvs(target_size, rtks.cpu().numpy(), 
                                    near_far, rndmask, model.device)
    
    # query env code
    rays['env_code'] = model.env_code(frameid)[:,None]
    rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

    # query deformation
    time_embedded = model.pose_code(frameid)[:,None]
    rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)

    if opts.use_separate_code_dfm:
        if opts.combine_dfm_and_pose:
            dfm_embedded = model.dfm_code_w_pose(frameid)[:, None]
        else:
            dfm_embedded = model.dfm_code(frameid)[:, None]
        rays['dfm_embedded'] = dfm_embedded.repeat(1,rays['nsample'],1)

    bone_rts = model.nerf_body_rts(frameid)
    rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
    model.update_delta_rts(rays)

    with torch.no_grad():
        # render images only
        results=defaultdict(list)
        bs_rays = rays['bs'] * rays['nsample'] #
        for j in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,j,opts.chunk)
            rendered_chunks = render_rays(nerf_models,
                        embeddings,
                        rays_chunk,
                        N_samples = opts.ndepth,
                        perturb=0,
                        noise_std=0,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        use_fine=True,
                        img_size=model.img_size,
                        obj_bound = model.latest_vars['obj_bound'],
                        render_vis=True,
                        opts=opts,
                        render_losses=False
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v.cpu()]

    for k, v in results.items():
        v = torch.cat(v, 0)
        v = v.view(rays['nsample'], -1)
        results[k] = v

    original_rgb = cv2.resize(elem["img"][0].transpose(1,2,0), (w,h)).reshape(h, w, 3)
    org_freq_shifted = np.fft.fftshift(np.fft.fft2(original_rgb, axes=(0,1)))
    original_sil = cv2.resize(elem["mask"][0], (w,h))
    rgb_coarse = results['img_coarse'].numpy().reshape(h, w, 3)
    rgb_coarse[original_sil<0.5] = 0
    has_fine = False
    if 'img_fine' in results.keys():
        rgb_fine = results['img_fine'].numpy().reshape(h, w, 3)
        rgb_fine[original_sil<0.5] = 0
        has_fine = True

    sil = results['sil_coarse'][...,0].numpy().reshape(h, w)
    original_rgb[original_sil<0.5] = 0
    
    cv2.imwrite(f"{path_name}/coarse_{frameid[0].item()}.jpg", (rgb_coarse*255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite(f"{path_name}/orig_{frameid[0].item()}.jpg", (original_rgb*255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite(f"{path_name}/silpred_{frameid[0].item()}.jpg", (sil*255).astype(np.uint8))
    
    # Eval ssim
    original_rgb_uint8 = (original_rgb*255).astype(np.uint8)
    rgb_coarse_uint8 = (rgb_coarse*255).astype(np.uint8)
    (score, diff) = compare_ssim(cv2.cvtColor(original_rgb_uint8, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb_coarse_uint8, cv2.COLOR_RGB2GRAY), full=True)
    img_ssim_mean_coarse.append(score)

    hp_filter = cv2.resize(gkern(kernlen=max(w,h), std=11).numpy(), (w, h))
    hp_filter[hp_filter<0.2] = 0
    org_ampl = org_freq_shifted.real**2 + org_freq_shifted.imag**2
    if has_fine:
        cv2.imwrite(f"{path_name}/fine_{frameid[0].item()}.jpg", (rgb_fine*255).astype(np.uint8)[:,:,::-1])
        syn_freq_shifted = np.fft.fftshift(np.fft.fft2(rgb_fine, axes=(0,1)))
        PSNRs.append(10 * np.log10(255**2 / (((original_rgb-rgb_fine)*255)**2).mean() ) )
        img_mean_mse.append( ((original_rgb-rgb_fine)**2).sum(-1).mean() )
        syn_ampl = syn_freq_shifted.real**2 + syn_freq_shifted.imag**2
        hf_amp_mean_mse.append( np.log((syn_ampl - org_ampl)**2 * hp_filter[:,:,np.newaxis] + 1).mean() )
        rgb_fine_uint8 = (rgb_fine*255).astype(np.uint8)
        (score, diff) = compare_ssim(cv2.cvtColor(original_rgb_uint8, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb_fine_uint8, cv2.COLOR_RGB2GRAY), full=True)
        img_ssim_mean.append(score)
    else:
        syn_freq_shifted = np.fft.fftshift(np.fft.fft2(rgb_coarse, axes=(0,1)))
        img_mean_mse.append( ((original_rgb-rgb_coarse)**2).sum(-1).mean() )
        syn_ampl = syn_freq_shifted.real**2 + syn_freq_shifted.imag**2
    
    syn_freq_shifted = np.fft.fftshift(np.fft.fft2(rgb_coarse, axes=(0,1)))
    PSNRs_coarse.append(10 * np.log10(255**2 / (((original_rgb-rgb_coarse)*255)**2).mean() ) )
    hf_amp_mean_mse_coarse.append( np.log((syn_ampl - org_ampl)**2 * hp_filter[:,:,np.newaxis] + 1).mean() )
    sil_mean_mse.append( ((original_sil-sil)**2).mean() )
    

  print("************************************************")
  print("------------------------------------------------")
  if has_fine:
    print("Mean fine PSNR:", np.array(PSNRs).mean())
    print("Mean fine SSIM:", np.array(img_ssim_mean).mean())
    print("Mean hf fine amplitude diff:", np.array(hf_amp_mean_mse).mean())
  print("Mean MSE:", np.array(img_mean_mse).mean())
  print("------------------------------------------------")
  print("Mean coarse PSNR:", np.array(PSNRs_coarse).mean())
  print("Mean coarse SSIM:", np.array(img_ssim_mean_coarse).mean())
  print("Mean hf coarse amplitude diff:", np.array(hf_amp_mean_mse_coarse).mean())
  print("Mean sil MSE:", np.array(sil_mean_mse).mean())
  print("************************************************")
  

if __name__ == '__main__':
    app.run(main)
