# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import flags, app
import sys
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from utils.io import get_vertex_colors
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, \
                               gauss_mlp_skinning, diff_flo

def deform_mesh_vertices(model, mesh, frameid, opts):
    # Coarse dfm
    models = model.nerf_models
    xyz_coarse_sampled = torch.cuda.FloatTensor(mesh.vertices, device=model.device).unsqueeze(0)

    # query deformation
    frameid = torch.Tensor([frameid]).to(model.device).long()
    rays = {}
    rays['time_embedded'] = model.pose_code(frameid)[:,None]
    
    if opts.use_separate_code_dfm:
        if opts.combine_dfm_and_pose:
            dfm_embedded = model.dfm_code_w_pose(frameid)
        else:
            dfm_embedded = model.dfm_code(frameid)

        rays['dfm_embedded'] = dfm_embedded
        
    bone_rts = model.nerf_body_rts(frameid)
    bones_rst = models['bones_rst']
    bone_rts_fw = bone_rts
    skin_aux = models['skin_aux']
    rest_pose_code =  models['rest_pose_code'] # To obtain pose code for specific bone
    rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device)) # Embedding of size 128
    
    if 'nerf_skin' in models.keys():
        # compute delta skinning weights of bs, N, B
        nerf_skin = models['nerf_skin'] 
    else:
        nerf_skin = None

    time_embedded = rays['time_embedded'] #[:,None] 
    # coords after deform
    bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True) # Bones after applying a forward (canonical -> bones at time t) rigid transformation
    skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, model.embedding_xyz, 
                bones_dfm, time_embedded,  nerf_skin, chunk_mult_eval_mlp=opts.chunk_mult_eval_mlp, skin_aux=skin_aux)

    # backward skinning -> Obtain NeRF samples in canonical space? Yes (also returning here bones_dfm for some reason when they have already been computed)
    xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                bone_rts_fw, 
                                                skin_backward,
                                                xyz_coarse_sampled,
                                                )

    if opts.fine_nerf_net and opts.extra_dfm_nr and opts.use_sdf_finenerf:
        # Apply fine deformation
        n_r, n_s = xyz_coarse_sampled.size(0), xyz_coarse_sampled.size(1)
        if not opts.use_separate_code_dfm:
            time_embedded = rays['time_embedded'] #[:,None]
            enc_temp = time_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, time_embedded.size(-1))
        else:
            dfm_embedded = rays['dfm_embedded'][:,None]
            enc_temp = dfm_embedded.expand((-1, n_s, -1)).resize(n_r*n_s, dfm_embedded.size(-1))
        
        xyz_coarse_sampled_no_def = xyz_coarse_sampled.resize(n_r*n_s, 3)
        enc_spat = model.embedding_xyz(xyz_coarse_sampled_no_def)
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

        xyz_coarse_sampled_def = xyz_coarse_sampled + pts_dfm.resize(n_r, n_s, 3)
    else:
        xyz_coarse_sampled_def = xyz_coarse_sampled

    return xyz_coarse_sampled_def.squeeze(0)

def save_output(model, rendered_seq, aux_seq, seqname, save_flo):
    with torch.no_grad():
        save_dir = '%s/'%(opts.model_path.rsplit('/',1)[0])
        length = len(aux_seq['mesh'])
        mesh_rest = aux_seq['mesh_rest']
        len_max = (mesh_rest.vertices.max(0) - mesh_rest.vertices.min(0)).max()
        # Already computeds with viewing dir correctllyylyy
        #if opts.noce_color:
        #    print("OBTAINING VERTEX COLORS")
        # vis = get_vertex_colors(model, mesh_rest, frame_idx=0, fine_nerf_net=opts.fine_nerf_net)
        # mesh_rest.visual.vertex_colors[:,:3] = vis*255
        mesh_rest.export('%s/mesh-rest.obj'%save_dir)
        mesh_rest.fill_holes() # Fill holes so that it is watertight
        mesh_rest.export('%s/mesh-rest-watertight.obj'%save_dir)
        if 'mesh_rest_skin' in aux_seq.keys():
            aux_seq['mesh_rest_skin'].export('%s/mesh-rest-skin.obj'%save_dir)
        if 'bone_rest' in aux_seq.keys():
            bone_rest = aux_seq['bone_rest']
            save_bones(bone_rest, len_max, '%s/bone-rest.obj'%save_dir)

        flo_gt_vid = []
        flo_p_vid = []
        for i in range(length):
            impath = aux_seq['impath'][i]
            seqname = impath.split('/')[-2]
            save_prefix = '%s/%s'%(save_dir,seqname)
            idx = int(impath.split('/')[-1].split('.')[-2])
            mesh = aux_seq['mesh'][i]
            rtk = aux_seq['rtk'][i]
            
            # convert bones to meshes TODO: warp with a function
            if 'bone' in aux_seq.keys() and len(aux_seq['bone'])>0:
                bones = aux_seq['bone'][i]
                bone_path = '%s-bone-%05d.obj'%(save_prefix, idx)
                save_bones(bones, len_max, bone_path)

            if opts.noce_color:
                print("OBTAINING VERTEX COLORS")
                view_dir = aux_seq['view_dir'][i]
                if idx != 0:
                    vertices_dfm = deform_mesh_vertices(model, mesh, idx, opts)
                    vis = get_vertex_colors(model, mesh, vertices=vertices_dfm, view_dir=view_dir, frame_idx=idx, fine_nerf_net=opts.fine_nerf_net)
                else:
                    vis = get_vertex_colors(model, mesh, frame_idx=idx, view_dir=view_dir, fine_nerf_net=opts.fine_nerf_net)
                mesh.visual.vertex_colors[:,:3] = vis*255

            mesh.export('%s-mesh-%05d.obj'%(save_prefix, idx))
            np.savetxt('%s-cam-%05d.txt'  %(save_prefix, idx), rtk)
            
            #try:
            img_gt = rendered_seq['img'][i]
            flo_gt = rendered_seq['flo'][i]
            mask_gt = rendered_seq['sil'][i][...,0]
            flo_gt[mask_gt<=0] = 0
            img_gt[mask_gt<=0] = 1
            if save_flo: img_gt = cat_imgflo(img_gt, flo_gt)
            else: img_gt*=255
            cv2.imwrite('%s-img-gt-%05d.jpg'%(save_prefix, idx), img_gt[...,::-1])
            flo_gt_vid.append(img_gt)
            
            img_p = rendered_seq['img_coarse'][i]
            flo_p = rendered_seq['flo_coarse'][i]
            mask_gt = cv2.resize(mask_gt, flo_p.shape[:2][::-1]).astype(bool)
            flo_p[mask_gt<=0] = 0
            img_p[mask_gt<=0] = 1
            if save_flo: img_p = cat_imgflo(img_p, flo_p)
            else: img_p*=255
            cv2.imwrite('%s-img-p-%05d.jpg'%(save_prefix, idx), img_p[...,::-1])
            flo_p_vid.append(img_p)

            flo_gt = cv2.resize(flo_gt, flo_p.shape[:2])
            flo_err = np.linalg.norm( flo_p - flo_gt ,2,-1)
            flo_err_med = np.median(flo_err[mask_gt])
            flo_err[~mask_gt] = 0.
            cv2.imwrite('%s-flo-err-%05d.jpg'%(save_prefix, idx), 
                    128*flo_err/flo_err_med)

            img_gt = rendered_seq['img'][i]
            img_p = rendered_seq['img_coarse'][i]
            img_gt = cv2.resize(img_gt, img_p.shape[:2][::-1])
            img_err = np.power(img_gt - img_p,2).sum(-1)
            img_err_med = np.median(img_err[mask_gt])
            img_err[~mask_gt] = 0.
            cv2.imwrite('%s-img-err-%05d.jpg'%(save_prefix, idx), 
                    128*img_err/img_err_med)
            #except:
            #    print("failed saving image")


        #    fps = 1./(5./len(flo_p_vid))
        try:
            upsample_frame = min(30, len(flo_p_vid))
            save_vid('%s-img-p' %(save_prefix), flo_p_vid, upsample_frame=upsample_frame)
            save_vid('%s-img-gt' %(save_prefix),flo_gt_vid,upsample_frame=upsample_frame)
        except:
            print("could not save video")

def transform_shape(mesh,rtk):
    """
    (deprecated): absorb rt into mesh vertices, 
    """
    vertices = torch.Tensor(mesh.vertices)
    Rmat = torch.Tensor(rtk[:3,:3])
    Tmat = torch.Tensor(rtk[:3,3])
    vertices = obj_to_cam(vertices, Rmat, Tmat)

    rtk[:3,:3] = np.eye(3)
    rtk[:3,3] = 0.
    mesh = trimesh.Trimesh(vertices.numpy(), mesh.faces)
    return mesh, rtk

def main(_):
    opts.finer_reconstruction = False
    opts.keep_original_mesh = True
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    seqname=opts.seqname

    dynamic_mesh = opts.flowbw or opts.lbs
    idx_render = str_to_frame(opts.test_frames, data_info)
#    idx_render[0] += 50
#    idx_render[0] += 374
#    idx_render[0] += 292
#    idx_render[0] += 10
#    idx_render[0] += 340
#    idx_render[0] += 440
#    idx_render[0] += 540
#    idx_render[0] += 640
#    idx_render[0] += trainer.model.data_offset[4]-4 + 37
#    idx_render[0] += 36

    print("total:", idx_render)
    trainer.model.img_size = opts.render_size
    chunk = opts.frame_chunk # TOADD: idx_render = np.array(idx_render)
    #idx_render = np.arange(idx_render[-1]) 
    #idx_render = np.arange(idx_render[1])
    for i in range(0, len(idx_render), chunk):
        rendered_seq, aux_seq = trainer.eval(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh, keep_original_mesh=opts.keep_original_mesh) 
        rendered_seq = tensor2array(rendered_seq)
        save_output(trainer.model, rendered_seq, aux_seq, seqname, save_flo=opts.use_corresp)
    #TODO merge the outputs

if __name__ == '__main__':
    app.run(main)
