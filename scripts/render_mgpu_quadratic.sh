# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#bash scripts/render_mgpu.sh 0 logdir/$seqname-ft3/params_latest.pth \
#        "0 1 2 3 4 5 6 7 8 9 10"'
## argv[1]: gpu id
## argv[2]: sequence name
## argv[3]: weights path
## argv[4]: video id separated by space
## argv[5]: resolution of running marching cubes (256 by default)

dev=$1
seqname=$2
modelpath=$3
vids=$4
sample_grid3d=$5

CUDA_VISIBLE_DEVICES=${dev} bash scripts/render_vids.sh \
  ${seqname} ${modelpath} "${vids}" \
  "--sample_grid3d ${sample_grid3d} \
  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2 --noce_color --photometric_loss l1 --bound_fine --bound_factor_fine 3 --only_coarse_points True --num_iters_small_training 400 --eval_every_n 20 --nerf_n_layers_c 8 --nerf_n_layers_f 8 --num_freqs 10 --fine_nerf_net --extra_dfm_nr --dfm_type quadratic --sil_wt 1 --img_wt 0.1 --wsd 1e-2 --nouse_window --nolarge_cond_embedding --nofine_samples_from_coarse --noone_cycle_lr --reset_lr --use_sdf_finenerf --use_separate_code_dfm --combine_dfm_and_pose --dfm_emb_dim 64 --render_size 32 --smoothness_dfm --smoothness_dfm_spat --train_always_unc --nsample_a_mult 16 --mult_img_wt_on_refine --img_wt_mlt 10 --weight_smoothness_opacity --neighborhood_scale 1e-2 --noqueryfw"
