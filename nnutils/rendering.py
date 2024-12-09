# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# adopted from nerf-pl
import numpy as np
import pdb
import torch
import torch.nn.functional as F
from pytorch3d import transforms

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, \
                               gauss_mlp_skinning, diff_flo, cast_rays_means_covs_cone, integrated_pos_enc
from nnutils.loss_utils import elastic_loss, visibility_loss, feat_match_loss,\
                                kp_reproj_loss, compute_pts_exp, kp_reproj, evaluate_mlp, photometric_loss, \
                                elastic_loss_mesh, elastic_loss_nerfies, elastic_loss_smooth_neighborhood_in_ray, \
                                apply_smoothness_type_loss

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                chunk=1024*32,
                obj_bound=None,
                use_fine=False,
                img_size=None,
                progress=None,
                opts=None,
                render_vis=False,
                render_losses=True
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    if use_fine and not opts.fine_nerf_net: N_samples = N_samples//2 # use half samples to importance sample
    # TODO: Correct nerf arch (no appearance code is being used!)
    # Extract models from lists
    embedding_xyz = embeddings['xyz']
    embedding_dir = embeddings['dir']

    # Decompose the inputs
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']  # both (N_rays, 3)
    near = rays['near']
    far = rays['far']  # both (N_rays, 1)
    N_rays = rays_d.shape[0]

    # Embed direction
    rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]
    dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)
    
    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # zvals are not optimized
    # produce points in the root body space
    xyz_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if not opts.fine_nerf_net:
        if use_fine: # sample points for fine model
            # output: 
            #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
            #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
            #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
            with torch.no_grad():
                _, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                                chunk, N_samples,
                                N_rays, embedding_xyz, rays_d, noise_std,
                                obj_bound, dir_embedded, z_vals,
                                img_size, progress,opts,extra_dfm_nr=opts.extra_dfm_nr, fine_iter=False, use_sdf=opts.use_sdf)

                # reset N_importance
                N_importance = N_samples #* 2
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
                z_means = z_vals_mid.mean(1)
            
            # Trick of VAEs as sampling is not differentiable
            if opts.sample_laplacian:
                with torch.no_grad():
                    lapl_dist = torch.distributions.laplace.Laplace(0, 1, validate_args=None)
                    z_vals_ = lapl_dist.sample((z_vals.size(0), N_importance)).to(device="cuda:0")*models["coarse"].beta + z_means.unsqueeze(1)
            else:
                z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                                N_importance, det=(perturb==0)).detach()

            #lapl_dists = [torch.distributions.laplace.Laplace(z_means[i], 1e-9 + 0.1*models["coarse"].beta, validate_args=None) for i in range(z_vals.size(0))]
            #z_vals_ = torch.cat([lapl_dists[i].sample((N_importance,)).transpose(1, 0) for i in range(len(lapl_dists))], 0)
            #z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
            #                    N_importance, det=(perturb==0)).detach()
                    # detach so that grad doesn't propogate to weights_coarse from here

            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

            xyz_sampled = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

            N_samples = N_samples + N_importance # get back to original # of samples
        
        # Deform NeRF sampled points?
        result, _ = inference_deform(xyz_sampled, rays, models, 
                            chunk, N_samples,
                            N_rays, embedding_xyz, rays_d, noise_std,
                            obj_bound, dir_embedded, z_vals,
                            img_size, progress,opts,fine_iter=render_losses, 
                            extra_dfm_nr=opts.extra_dfm_nr and use_fine, render_vis=render_vis, 
                            use_sdf=opts.use_sdf)

    else:
        # Deform NeRF sampled points?
        if not opts.only_coarse_points:
            result, _ = inference_deform(xyz_sampled, rays, models, 
                            chunk, N_samples,
                            N_rays, embedding_xyz, rays_d, noise_std,
                            obj_bound, dir_embedded, z_vals,
                            img_size, progress,opts, fine_iter=render_losses, 
                            extra_dfm_nr=False, render_vis=render_vis,use_sdf=opts.use_sdf)
            
            # extra_dfm_nr=opts.extra_dfm_nr
            #if use_fine: # sample points for fine model # Maybe always train fine model

            # output: 
            #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
            #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
            #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
            use_sdf = opts.use_sdf
            extra_dfm_nr = False
            model_coarse = models["coarse"]
            models["coarse"] = models["fine"]
            use_sdf = False
            extra_dfm_nr = opts.extra_dfm_nr

            if opts.use_sdf_finenerf:
                use_sdf = opts.use_sdf_finenerf

            with torch.no_grad():
                res_s, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                                chunk, N_samples,
                                N_rays, embedding_xyz, rays_d, noise_std,
                                obj_bound, dir_embedded, z_vals,
                                img_size, progress,opts,extra_dfm_nr=extra_dfm_nr, 
                                is_fine_nerf=True, 
                                fine_iter=False, use_sdf=use_sdf, use_IPE=opts.IPE)

                # reset N_importance
                N_importance = N_samples #* 2
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
                z_means = z_vals_mid.mean(1)
            
            # Trick of VAEs as sampling is not differentiable
            if opts.sample_laplacian:
                with torch.no_grad():
                    lapl_dist = torch.distributions.laplace.Laplace(0, 1, validate_args=None)
                    z_vals_ = lapl_dist.sample((z_vals.size(0), N_importance)).to(device="cuda:0")*models["coarse"].beta + z_means.unsqueeze(1)
            else:
                z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1].detach(),
                                N_importance, det=(perturb==0)).detach()

            if not opts.fine_samples_from_coarse: # If fine samples coming from fine net, then include original ones (so that weights make sense)
                N_samples = N_importance * 2
                z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
            else:
                N_samples = N_importance # get back to original # of samples
                z_vals = z_vals_

            xyz_sampled = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

            if not use_fine:
                extra_dfm_nr = False
            else:
                extra_dfm_nr = opts.extra_dfm_nr
            obj_bound_fine = obj_bound

            if opts.bound_fine: # (Arbitrarily) Triple the bound size for fine nerf (how to set this?)
                obj_bound_fine = obj_bound_fine * opts.bound_factor_fine
            # 
            result2, _ = inference_deform(xyz_sampled, rays, models, 
                                chunk, N_samples,
                                N_rays, embedding_xyz, rays_d, noise_std,
                                obj_bound_fine, dir_embedded, z_vals,
                                img_size, progress, opts, fine_iter=render_losses, is_fine_nerf=True,
                                extra_dfm_nr=opts.extra_dfm_nr, render_vis=render_vis,
                                use_sdf=opts.use_sdf_finenerf, use_IPE=opts.IPE)

            fine_losses = ["extra_dfm_loss", "smoothness_dfm_loss", "smoothness_dfm_temp_loss", "img_loss_samp", "img_coarse", "sil_coarse", "sil_loss_samp"]
            result_name = ["extra_dfm_loss", "smoothness_dfm_loss", "smoothness_dfm_temp_loss", "img_loss_samp_fine", "img_fine", "sil_fine", "sil_loss_samp_fine"]

            for i in range(len(fine_losses)):
                if fine_losses[i] in result2.keys():
                    result[result_name[i]] = result2[fine_losses[i]]    

            if "vis_loss" in result2.keys() and opts.use_sdf_finenerf:
                result["vis_loss"] = result2["vis_loss"] 
            
            # Additional ones
            if "proj_err" in result.keys():
                #if opts.warmup_steps > 0.01:
                #    result["flo_loss_samp"] = result["flo_loss_samp"] + result2["flo_loss_samp"]
                #    result["proj_err"] = result["proj_err"] + result2["proj_err"]
                #result["feat_err"] = result["feat_err"] + result2["feat_err"]
                result["frnd_loss_samp"] = result["frnd_loss_samp"] + result2["frnd_loss_samp"]

            #if "sdf" in result.keys() and "sdf" in res_s.keys():
            #    result["sdf_loss"] = ((result["sdf"].detach() - res_s["sdf"])**2).mean()

            # sil_loss_fine
            models["fine"] = models["coarse"]
            models["coarse"] = model_coarse
        else:
            
            model_coarse = models["coarse"]
            models["coarse"] = models["fine"]
            use_sdf = False
            extra_dfm_nr = opts.extra_dfm_nr

            if opts.use_sdf_finenerf:
                use_sdf = opts.use_sdf_finenerf

            result, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                            chunk, N_samples,
                            N_rays, embedding_xyz, rays_d, noise_std,
                            obj_bound, dir_embedded, z_vals,
                            img_size, progress,opts,extra_dfm_nr=extra_dfm_nr, 
                            is_fine_nerf=True, 
                            fine_iter=render_losses, use_sdf=use_sdf, use_IPE=opts.IPE)
                
            models["fine"] = models["coarse"]
            models["coarse"] = model_coarse

    return result
    
def inference(models, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, weights_only=False, clip_bound = None, vis_pred=None, 
        is_fine_nerf=False, use_sdf=True, rays_radii=None, rays_origin=None, use_IPE=False):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        rgb_final: (N_rays, 3) the final rgb image
        depth_final: (N_rays) depth map
        weights: (N_rays, N_samples_): weights of each sample
    """
    # ###
    nerf_sdf = models['coarse']
    N_samples_ = xyz_.shape[1]
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    chunk_size=4096
    B = xyz_.shape[0]
    # Embed directions
    if not weights_only:
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                       # (N_rays*N_samples_, embed_dir_channels)

    # Get IPE
    if use_IPE and not rays_radii is None:
        means_covs = cast_rays_means_covs_cone(z_vals, rays_origin, dir_, rays_radii) 
        nerf_rgb_xyz_ = integrated_pos_enc(
            means_covs,
            0,
            16,
        )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)
        nerf_rgb_xyz_ = torch.cat([nerf_rgb_xyz_, nerf_rgb_xyz_[:,-2:-1,:]], 1)
        out = evaluate_mlp(nerf_sdf, nerf_rgb_xyz_, # Evaluate MLP of NeRF doing embedding in position/direction and chunking data into batches
                embed_xyz = None,
                dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
                code=env_code, # env code is something related to a specific view I think like illumination conditions code
                chunk=chunk_size, sigma_only=weights_only).view(B,-1)
        xyz_input = xyz_.view(N_rays,N_samples,3)
    else:
        # Perform model inference to get rgb and raw sigma
        xyz_input = xyz_.view(N_rays,N_samples,3)
        out = evaluate_mlp(nerf_sdf, xyz_input, # Evaluate MLP of NeRF doing embedding in position/direction and chunking data into batches
                embed_xyz = embedding_xyz,
                dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
                code=env_code, # env code is something related to a specific view I think like illumination conditions code
                chunk=chunk_size, sigma_only=weights_only).view(B,-1)

    rgbsigma = out.view(N_rays, N_samples_, out.size(-1))
    rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
    sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

    if 'nerf_feat' in models.keys(): # NeRF feat for something of the CSE
        nerf_feat = models['nerf_feat']
        feat = evaluate_mlp(nerf_feat, xyz_input,
            embed_xyz = embedding_xyz,
            chunk=chunk_size).view(N_rays,N_samples_,-1)
    else:
        feat = torch.zeros_like(rgbs)

    # Convert these values using volume rendering (Section 3.1) -> Laplace distribution for converting sdf to density
    if use_sdf:
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        # a hacky way to ensures prob. sum up to 1     
        # while the prob. of last bin does not correspond with the values
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        sigmas = sigmas+noise
        #sigmas = F.softplus(sigmas)
        #sigmas = torch.relu(sigmas)
        ibetas = 1/(nerf_sdf.beta.abs()+1e-9)
        #ibetas = 100
        sdf = -sigmas
        sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
        # alternative: 
        #sigmas = F.sigmoid(-sdf*ibetas)
        sigmas = sigmas * ibetas
        alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

        #set out-of-bound and nonvisible alphas to zero
        #if not is_fine_nerf:
        if clip_bound is not None:
            clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
            oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
            alphas[oob]=0
        if vis_pred is not None:
            alphas[vis_pred<0.5] = 0

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
        weights = alphas * alpha_prod # (N_rays, N_samples_)
        #if is_fine_nerf:
        #    act_sigmas = torch.log(1+torch.exp(weights))
        #    weights = act_sigmas / act_sigmas.sum(1).unsqueeze(1)

    else:
        alpha_prod = torch.sigmoid(sigmas) #.unsqueeze(0)
        act_sigmas = torch.log(1+torch.exp(alpha_prod)) + 1e-5
        weights = act_sigmas / act_sigmas.sum(1).unsqueeze(1)
    
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1 # TODO: Visibility how? Maybe just ignore? Use visibility of coarse and not predicting it on the fine one

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    feat_final = torch.sum(weights.unsqueeze(-1)*feat, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    return rgb_final, feat_final, depth_final, weights, visibility, sdf
    
def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress,opts, extra_dfm_nr=False, fine_iter=True, 
                         render_vis=False, is_fine_nerf=False, use_sdf=True, use_IPE=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """

    is_training = models['coarse'].training
    xys = rays['xys']
    # NeRF xyz_coarse_sampled are to be rendered in the space at time t (not the canonical space)
    # root space point correspondence in t2
    if opts.dist_corresp:
        xyz_coarse_target = xyz_coarse_sampled.clone()
        xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()

    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
        
        if fine_iter:
            # cycle loss (in the joint canonical space)
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # rigidity loss
            frame_disp3d = flow_fw.norm(2,-1)

            if "time_embedded_target" in rays.keys():
                time_embedded_target = rays['time_embedded_target'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            if "time_embedded_dentrg" in rays.keys():
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw

    elif 'bones' in models.keys():
        bones_rst = models['bones_rst']
        bone_rts_fw = rays['bone_rts']
        skin_aux = models['skin_aux']
        rest_pose_code =  models['rest_pose_code'] # To obtain pose code for specific bone
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device)) # Embedding of size 128
        
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        time_embedded = rays['time_embedded'][:,None] # ?
        # coords after deform
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True) # Bones after applying a forward (canonical -> bones at time t) rigid transformation
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                    bones_dfm, time_embedded,  nerf_skin, chunk_mult_eval_mlp=opts.chunk_mult_eval_mlp, skin_aux=skin_aux)

        # backward skinning -> Obtain NeRF samples in canonical space? Yes (also returning here bones_dfm for some reason when they have already been computed)
        xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )

        if fine_iter:
            # Skinning weights in canonical space from deformed points (backward transformed points from frame t to canonical position)
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_rst,rest_pose_code,  nerf_skin, chunk_mult_eval_mlp=opts.chunk_mult_eval_mlp, skin_aux=skin_aux)

            # cycle loss (in the joint canonical space) -> Points backwarded based on skinning weights, forwarded and compared with l2 distance
            xyz_coarse_frame_cyc,_ = lbs(bones_rst, bone_rts_fw,
                              skin_forward, xyz_coarse_sampled, backward=False)
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
            
            # rigidity loss (not used as optimization objective) -> Used to measure translation / rotation of bone -> The greater the less rigid
            num_bone = bones_rst.shape[0] 
            bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
            bone_trn = bone_fw_reshape[:,:,9:12]
            bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
            frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
            # Two more different rigid transformations (bone_rts_target and bone_rts_dentrg), not sure why. Applied to forward skinning in points in canonical frame
            if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                bone_rts_target = rays['bone_rts_target']
                xyz_coarse_target,_ = lbs(bones_rst, bone_rts_target, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                bone_rts_dentrg = rays['bone_rts_dentrg']
                xyz_coarse_dentrg,_ = lbs(bones_rst, bone_rts_dentrg, 
                                   skin_forward, xyz_coarse_sampled,backward=False)

    # Extra deformation
    if extra_dfm_nr:
        n_r, n_s = xyz_coarse_sampled.size(0), xyz_coarse_sampled.size(1)
        if not opts.use_separate_code_dfm:
            time_embedded = rays['time_embedded'][:,None]
            enc_temp = time_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, time_embedded.size(-1))
        else:
            dfm_embedded = rays['dfm_embedded'][:,None]
            enc_temp = dfm_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded.size(-1))
        
        xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
        enc_spat = embedding_xyz(xyz_coarse_sampled_no_def)
        # enc_temp = torch.cat([embedding.expand((vertices.shape[0]//embed_id.size(0), -1)) for embedding in enc_temp], 0)
        mesh_def_encoding = torch.cat([enc_spat, enc_temp], 1)
        pts_dfm = models["extra_deform_net"](mesh_def_encoding) # otherwise at start they are too big
        if opts.dfm_type == "quadratic":
            quad_params = pts_dfm.resize(pts_dfm.size(0), 3, 9)
            coords_quad = xyz_coarse_sampled.resize(n_r*n_s, 1, 3)
            coords_quad = torch.cat([coords_quad, coords_quad**2, 
                            (coords_quad[:,:,0]*coords_quad[:,:,1]).unsqueeze(-1), 
                            (coords_quad[:,:,1]*coords_quad[:,:,2]).unsqueeze(-1), 
                            (coords_quad[:,:,0]*coords_quad[:,:,2]).unsqueeze(-1)], 2)
            coords_quad = coords_quad.repeat(1, 3, 1)
            pts_dfm = (quad_params*coords_quad).sum(2)

        """ if opts.smoothness_dfm_temp:
            dfm_embedded_next = rays['time_embedded_next'][:,None]
            enc_temp_next = dfm_embedded_next.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded_next.size(-1))
            mesh_def_encoding_temp = torch.cat([enc_spat, enc_temp_next], 1)
            pts_dfm_temp = models["extra_deform_net"](mesh_def_encoding_temp) """

        xyz_coarse_sampled_def = xyz_coarse_sampled + pts_dfm.resize(n_r, n_s, 3)
    else:
        xyz_coarse_sampled_def = xyz_coarse_sampled
    
    # nerf shape/rgb
    model_coarse = models['coarse']
    if 'env_code' in rays.keys():
        env_code = rays['env_code']
    else:
        env_code = None

    # set out of bounds weights to zero ?
    if render_vis: 
        clip_bound = obj_bound
        xyz_embedded = embedding_xyz(xyz_coarse_sampled_def)
        vis_pred = evaluate_mlp(models['nerf_vis'], 
                               xyz_embedded, chunk=chunk)[...,0].sigmoid()
    else:
        clip_bound = None
        vis_pred = None

    if opts.symm_shape:
        ##TODO set to x-symmetric here
        symm_ratio = 0.5
        xyz_x = xyz_coarse_sampled_def[...,:1].clone()
        symm_mask = torch.rand_like(xyz_x) < symm_ratio
        xyz_x[symm_mask] = -xyz_x[symm_mask]
        xyz_input = torch.cat([xyz_x, xyz_coarse_sampled_def[...,1:3]],-1)
    else:
        xyz_input = xyz_coarse_sampled_def

    if opts.IPE:
        rays_radii = rays["radii"]
        rays_origin = rays["rays_o"]
    else:
        rays_radii = None
        rays_origin = None

    # Here extract coarse RGB and silhouette
    rgb_coarse, feat_rnd, depth_rnd, weights_coarse, vis_coarse, sdf = \
        inference(models, embedding_xyz, xyz_input, rays_d,
                dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
                weights_only=False, env_code=env_code, 
                clip_bound=clip_bound, vis_pred=vis_pred, is_fine_nerf=is_fine_nerf, use_sdf=use_sdf, 
                rays_radii=rays_radii,rays_origin=rays_origin, use_IPE=use_IPE)
    sil_coarse =  weights_coarse[:,:-1].sum(1) # sil_coarse is of shape (n_rays,)
    result = {'img_coarse': rgb_coarse,
              'depth_rnd': depth_rnd,
              'sil_coarse': sil_coarse
            }
    
    if models["nerf_vis"].training:
        result['weights_coarse'] = weights_coarse
        result['sdf'] = sdf

    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)

    if fine_iter: # Whether to render loss-related terms
        if opts.use_corresp: 
            # for flow rendering
            pts_exp = compute_pts_exp(weights_coarse, xyz_coarse_sampled_def) # Expected 3D point based on density (eq )
            pts_target = kp_reproj(pts_exp, models, embedding_xyz, rays,  # Reprojected points (expected 3D -> 2D) in another frame
                                to_target=True) # N,1,2
        # viser feature matching
        if 'feats_at_samp' in rays.keys(): # Feature matching loss (expected 3D point CSE close to reference vertex in CSE 3D mesh)
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            xyz_coarse_sampled_feat = xyz_coarse_sampled # Use non extra deformed thing for viser features (want extra deformation for higher frequencies)
            weights_coarse_feat = weights_coarse
            pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz,
                       feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                       obj_bound, is_training=is_training)

            # 3d-2d projection -> eq?
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays)
            proj_err = proj_err/img_size * 2
            
            result['pts_pred'] = pts_pred
            result['pts_exp']  = pts_exp
            result['feat_err'] = feat_err # will be used as loss -> Equation 15
            result['proj_err'] = proj_err # will be used as loss -> Equation ?
        # rtk_vec_target and rtk_vec_dentrg are for transforming sampled points in a ray to the coordinate system of another camera
        if opts.dist_corresp and 'rtk_vec_target' in rays.keys(): 
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)

        if opts.dist_corresp and 'rtk_vec_dentrg' in rays.keys():
            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat) 
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
        
        # raw 3d points for visualization
        result['xyz_camera_vis']   = xyz_coarse_frame 
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['xyz_canonical_vis']   = xyz_coarse_sampled_def
        if 'feats_at_samp' in rays.keys():
            result['pts_exp_vis']   = pts_exp
            result['pts_pred_vis']   = pts_pred
            
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            # cycle loss (in the joint canonical space)
            if opts.dist_corresp:
                result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
                #else:
                #    pts_exp_reg = pts_exp[:,None].detach()
                #    skin_forward = gauss_mlp_skinning(pts_exp_reg, embedding_xyz, 
                #                bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)
                #    pts_exp_fw,_ = lbs(bones_rst, bone_rts_fw,
                #                      skin_forward, pts_exp_reg, backward=False)
                #    skin_backward = gauss_mlp_skinning(pts_exp_fw, embedding_xyz, 
                #                bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)
                #    pts_exp_fwbw,_ = lbs(bones_rst, bone_rts_fw,
                #                       skin_backward,pts_exp_fw)
                #    frame_cyc_dis = (pts_exp_fwbw - pts_exp_reg).norm(2,-1)
                #    result['frame_cyc_dis'] = sil_coarse.detach() * frame_cyc_dis[...,-1]
                if 'flowbw' in models.keys():
                    result['frame_rigloss'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
                    # only evaluate at with_grad mode
                    if xyz_coarse_frame.requires_grad:
                        # elastic energy
                        result['elastic_loss'] = elastic_loss(model_flowbw, embedding_xyz, 
                                        xyz_coarse_frame, time_embedded)
                else:
                    result['frame_rigloss'] =  (frame_rigloss).mean(-1)

        if extra_dfm_nr and is_training:
            
            if opts.dfm_type == "nrnerf":
                extra_elastic_loss = elastic_loss_mesh(n_r, xyz_coarse_sampled_no_def, pts_dfm).mean()
            elif opts.dfm_type == "nrnerf+l2":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l2")
                extra_elastic_loss = extra_elastic_loss + elastic_loss_mesh(n_r, xyz_coarse_sampled_no_def, pts_dfm).mean()
            elif opts.dfm_type == "nerfies":
                extra_elastic_loss = elastic_loss_nerfies(n_r, xyz_coarse_sampled_no_def, pts_dfm)
            elif opts.dfm_type == "nerfies+l1":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l1")
                if opts.warmup_steps <= 0.1:
                    extra_elastic_loss = extra_elastic_loss + elastic_loss_nerfies(n_r, xyz_coarse_sampled_no_def, pts_dfm)
            elif opts.dfm_type == "nerfies+l2":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l2")
                if opts.warmup_steps <= 0.1:
                    extra_elastic_loss = extra_elastic_loss + elastic_loss_nerfies(n_r, xyz_coarse_sampled_no_def, pts_dfm)
            elif opts.dfm_type == "l1":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l1")
            elif opts.dfm_type == "l2":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l2")
            elif opts.dfm_type == "elasticnet":
                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="elasticnet")
            elif opts.dfm_type == "quadratic": # Smoothness in parameters of neighbor rays (IF CONSECUTIVE ADD EXTRA)
                extra_elastic_loss_spat = 0
                if opts.smoothness_dfm:
                    quad_params = quad_params.resize(quad_params.size(0), quad_params.size(1)*quad_params.size(2))
                    quad_params_smoothness = quad_params.resize(n_r, n_s, 27)
                    extra_elastic_loss_spat = apply_smoothness_type_loss(quad_params_smoothness[:,1:]-quad_params_smoothness[:,:-1], smoothness_type="l1")

                    if opts.weight_smoothness_opacity:
                        extra_elastic_loss_spat = (extra_elastic_loss_spat.flatten() * ((weights_coarse[:, 1:]+weights_coarse[:, :-1])/2).flatten()).sum(-1).mean()
                    else:
                        extra_elastic_loss_spat = extra_elastic_loss_spat.mean()

                extra_elastic_loss = apply_smoothness_type_loss(pts_dfm, smoothness_type="l2")

                if opts.smoothness_dfm_spat:
                    mean_diff = (pts_dfm**2).sum(-1).mean().detach() * opts.neighborhood_scale
                    n_random_neighbors = opts.n_random_neighbors
                    xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
                    neighbors_noise = torch.randn((n_random_neighbors, pts_dfm.size(0), pts_dfm.size(1)), device=mean_diff.device)*mean_diff
                    neighbors = xyz_coarse_sampled_no_def.unsqueeze(0)+neighbors_noise
                    neighbors = neighbors.resize(n_random_neighbors*pts_dfm.size(0), pts_dfm.size(1))
                    enc_spat = embedding_xyz(neighbors)
                    if not opts.use_separate_code_dfm:
                        time_embedded = rays['time_embedded'][:,None]
                        dfm_embedded_next = time_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, time_embedded.size(-1))
                    else:
                        dfm_embedded = rays['dfm_embedded'][:,None]
                        dfm_embedded_next = dfm_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded.size(-1))
                    #dfm_embedded_next = rays['time_embedded_next'][:,None]
                    enc_temp_current = dfm_embedded_next \
                                        .unsqueeze(0).repeat(n_random_neighbors, 1, 1) \
                                        .resize(n_random_neighbors*n_r*n_s, dfm_embedded_next.size(1))
                    mesh_def_encoding_temp = torch.cat([enc_spat, enc_temp_current], 1)
                    quad_params_temp = models["extra_deform_net"](mesh_def_encoding_temp)
                    quad_params_temp = quad_params_temp.resize(n_random_neighbors, n_r*n_s, 27)
                    smoothness_dfm_spat_loss = apply_smoothness_type_loss(quad_params_temp-quad_params.unsqueeze(0), smoothness_type="l2")
                    if opts.weight_smoothness_opacity:
                        smoothness_dfm_spat_loss = (smoothness_dfm_spat_loss * weights_coarse.flatten().unsqueeze(0)).resize(n_random_neighbors, n_r, n_s).sum(-1).mean()
                    else:
                        smoothness_dfm_spat_loss = smoothness_dfm_spat_loss.mean()
                    
                    result['smoothness_spat_dfm_loss'] = smoothness_dfm_spat_loss + extra_elastic_loss_spat

                if opts.smoothness_dfm_temp:
                    mean_diff = (pts_dfm**2).sum(-1).mean().detach() * opts.neighborhood_scale
                    n_random_neighbors = opts.n_random_neighbors
                    xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
                    neighbors_noise = torch.randn((n_random_neighbors, pts_dfm.size(0), pts_dfm.size(1)), device=mean_diff.device)*mean_diff
                    neighbors = xyz_coarse_sampled_no_def.unsqueeze(0)+neighbors_noise
                    neighbors = neighbors.resize(n_random_neighbors*pts_dfm.size(0), pts_dfm.size(1))
                    enc_spat = embedding_xyz(neighbors)
                    dfm_embedded_next = rays['time_embedded_next'][:,None]
                    enc_temp_next = dfm_embedded_next.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded_next.size(-1)) \
                                        .unsqueeze(0).repeat(n_random_neighbors, 1, 1) \
                                        .resize(n_random_neighbors*n_r*n_s, dfm_embedded_next.size(2))
                    mesh_def_encoding_temp = torch.cat([enc_spat, enc_temp_next], 1)
                    quad_params_temp = models["extra_deform_net"](mesh_def_encoding_temp)
                    quad_params_temp = quad_params_temp.resize(n_random_neighbors, n_r*n_s, 27)
                    smoothness_dfm_temp_loss = apply_smoothness_type_loss(quad_params_temp-quad_params.unsqueeze(0), smoothness_type="l2")
                    if opts.weight_smoothness_opacity:
                        smoothness_dfm_temp_loss = (smoothness_dfm_temp_loss * weights_coarse.flatten().unsqueeze(0)).resize(n_random_neighbors, n_r, n_s).sum(-1).mean()
                    else:
                        smoothness_dfm_temp_loss = smoothness_dfm_temp_loss.mean()
                    result['smoothness_dfm_temp_loss'] = smoothness_dfm_temp_loss + smoothness_dfm_temp_loss
                
            else:
                raise ValueError("Unknown deformation")

            if opts.smoothness_dfm:
                if opts.weight_smoothness_opacity and extra_elastic_loss.ndim > 0:
                    extra_elastic_loss = extra_elastic_loss * weights_coarse.flatten()
                result['extra_dfm_loss'] = extra_elastic_loss.mean()

            result['smoothness_dfm_loss'] = torch.Tensor(0).to(xyz_coarse_sampled.get_device())
            if opts.smoothness_dfm:
                result['smoothness_dfm_loss'] = elastic_loss_smooth_neighborhood_in_ray(pts_dfm.resize(n_r, n_s, 3))
                # .mean()
                if opts.weight_smoothness_opacity:
                    result['smoothness_dfm_loss'] = (result['smoothness_dfm_loss'].flatten() * ((weights_coarse[:, 1:]+weights_coarse[:, :-1])/2).flatten()).sum(-1).mean()
                else:
                    result['smoothness_dfm_loss'] = result['smoothness_dfm_loss'].mean()

            if "smoothness_spat_dfm_loss" in result.keys():
                result['smoothness_dfm_loss'] = result['smoothness_dfm_loss'] + result["smoothness_spat_dfm_loss"]

            if opts.smoothness_dfm_spat:
                if opts.dfm_type != "quadratic":
                    mean_diff = (pts_dfm**2).sum(-1).mean() * opts.neighborhood_scale
                    n_random_neighbors = opts.n_random_neighbors
                    xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
                    neighbors_noise = torch.randn((n_random_neighbors, pts_dfm.size(0), pts_dfm.size(1)), device=mean_diff.device)*mean_diff
                    neighbors = xyz_coarse_sampled_no_def.unsqueeze(0)+neighbors_noise
                    neighbors = neighbors.resize(n_random_neighbors*pts_dfm.size(0), pts_dfm.size(1))
                    enc_spat = embedding_xyz(neighbors)
                    if not opts.use_separate_code_dfm:
                        time_embedded = rays['time_embedded'][:,None]
                        dfm_embedded_next = time_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, time_embedded.size(-1))
                    else:
                        dfm_embedded = rays['dfm_embedded'][:,None]
                        dfm_embedded_next = dfm_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded.size(-1))
                    #dfm_embedded_next = rays['time_embedded_next'][:,None]
                    enc_temp_current = dfm_embedded_next \
                                        .unsqueeze(0).repeat(n_random_neighbors, 1, 1) \
                                        .resize(n_random_neighbors*n_r*n_s, dfm_embedded_next.size(1))
                    mesh_def_encoding_spat = torch.cat([enc_spat, enc_temp_current], 1)
                    pts_dfm_spat = models["extra_deform_net"](mesh_def_encoding_spat)
                    pts_dfm_spat = pts_dfm_spat.resize(n_random_neighbors, n_r*n_s, 3)
                    pts_dfm_spat_loss = apply_smoothness_type_loss(pts_dfm_spat-pts_dfm.unsqueeze(0), smoothness_type="l2")
                    if opts.weight_smoothness_opacity:
                        pts_dfm_spat_loss = (pts_dfm_spat_loss * weights_coarse.flatten().unsqueeze(0)).resize(n_random_neighbors, n_r, n_s).sum(-1).mean()
                    else:
                        pts_dfm_spat_loss = pts_dfm_spat_loss.mean()
                    result['smoothness_dfm_loss'] = result['smoothness_dfm_loss'] + pts_dfm_spat_loss

            if opts.smoothness_dfm_temp:
                if opts.dfm_type != "quadratic":
                    mean_diff = (pts_dfm**2).sum(-1).mean() * opts.neighborhood_scale
                    n_random_neighbors = opts.n_random_neighbors
                    xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
                    neighbors_noise = torch.randn((n_random_neighbors, pts_dfm.size(0), pts_dfm.size(1)), device=mean_diff.device)*mean_diff
                    neighbors = xyz_coarse_sampled_no_def.unsqueeze(0)+neighbors_noise
                    neighbors = neighbors.resize(n_random_neighbors*pts_dfm.size(0), pts_dfm.size(1))
                    enc_spat = embedding_xyz(neighbors)
                    dfm_embedded_next = rays['time_embedded_next'][:,None]
                    enc_temp_next = dfm_embedded_next.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded_next.size(-1)) \
                                        .unsqueeze(0).repeat(n_random_neighbors, 1, 1) \
                                        .resize(n_random_neighbors*n_r*n_s, dfm_embedded_next.size(2))
                    mesh_def_encoding_temp = torch.cat([enc_spat, enc_temp_next], 1)
                    pts_dfm_temp = models["extra_deform_net"](mesh_def_encoding_temp)
                    pts_dfm_temp = pts_dfm_temp.resize(n_random_neighbors, n_r*n_s, 3)
                    pts_dfm_temp_loss = apply_smoothness_type_loss(pts_dfm_temp-pts_dfm.unsqueeze(0), smoothness_type="l2")
                    if opts.weight_smoothness_opacity:
                        pts_dfm_temp_loss = (pts_dfm_temp_loss * weights_coarse.flatten().unsqueeze(0)).resize(n_random_neighbors, n_r, n_s).sum(-1).mean()
                    else:
                        pts_dfm_temp_loss = pts_dfm_temp_loss.mean()
                    result['smoothness_dfm_temp_loss'] = pts_dfm_temp_loss

        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled_def, vis_coarse, obj_bound, chunk)

        # render flow 
        if 'rtk_vec_target' in rays.keys(): # Optical flow computation using transformed sampled points in the ray to "target" image
            #if opts.dist_corresp:
            flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                xys, img_size)
            #else:
            #    flo_coarse = diff_flo(pts_target, xys, img_size)
            #    flo_valid = torch.ones_like(flo_coarse[...,:1])

            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

        if 'rtk_vec_dentrg' in rays.keys():
            #if opts.dist_corresp:
            fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                xyz_coarse_dentrg, xys, img_size)
            #else:
            #    fdp_coarse = diff_flo(pts_dentrg, xys, img_size)
            #    fdp_valid = torch.ones_like(fdp_coarse[...,:1])
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid

        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            #unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        
        if 'img_at_samp' in rays.keys(): # "img_at_samp" are the GT pixel values of the image
            # compute other losses
            img_at_samp = rays['img_at_samp']
            sil_at_samp = rays['sil_at_samp']
            vis_at_samp = rays['vis_at_samp']
            flo_at_samp = rays['flo_at_samp']
            cfd_at_samp = rays['cfd_at_samp']

            # Photometric loss: L1/L2/Hubber
            img_loss_samp = photometric_loss(opts.photometric_loss, rgb_coarse, img_at_samp) #(rgb_coarse - img_at_samp).pow(2).mean(-1)[...,None]
            
            # sil loss, weight sil loss based on # points -> Based in #points inside/outside the silhouette
            if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                pos_wt = vis_at_samp.sum()/   sil_at_samp[vis_at_samp>0].sum()
                neg_wt = vis_at_samp.sum()/(1-sil_at_samp[vis_at_samp>0]).sum()
                sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
            else: sil_balance_wt = 1
            sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
            sil_loss_samp = sil_loss_samp * vis_at_samp
            
            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1) # L2 diff optical flow
            # hard-threshold cycle error -> Account only for those optical flow values that are visible in the loss
            sil_at_samp_flo = (sil_at_samp>0)\
                     & (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            if sil_at_samp_flo.sum()>0:
                cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
            
            result['img_at_samp']   = img_at_samp
            result['sil_at_samp']   = sil_at_samp
            result['vis_at_samp']   = vis_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp_no_sil'] = img_loss_samp.clone()
            result['img_loss_samp'] = img_loss_samp 
            result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp
    
            # exclude error outside mask
            result['img_loss_samp']*=sil_at_samp
            result['flo_loss_samp']*=sil_at_samp

            if torch.isnan(result["img_loss_samp"]).any() or torch.any(result["img_loss_samp"] == torch.inf):
                print("Going out of bounds nan")
                #pts_dfm_reshaped = pts_dfm.resize(n_r * n_s, 3)
                #print("Bounds for fine are: min-", pts_dfm_reshaped.min(0)[0], "max-", pts_dfm_reshaped.max(0)[0])

        if 'feats_at_samp' in rays.keys():
            # feat loss
            feats_at_samp=rays['feats_at_samp']
            feat_rnd = F.normalize(feat_rnd, 2,-1)
            frnd_loss_samp = (feat_rnd - feats_at_samp).pow(2).mean(-1) # L2 loss CSE rendered feature and GT
            result['frnd_loss_samp'] = frnd_loss_samp #* sil_at_samp[...,0]
    return result, weights_coarse


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

