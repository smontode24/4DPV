<br>
<p align="center">
<h1 align="center"><strong>[ACCV2024] 4DPV: 4D Pet from Videos by Coarse-to-Fine Non-Rigid Radiance Fields</strong></h1>
  <p align="center">
    Sergio M. de Paco &emsp;
    Antonio Agudo&emsp;
    <br>
    <em>Institut de Robòtica i Informàtica Industrial CSIC-UPC, Barcelona, Spain</em>
    <br>
  </p>
</p>

We present a coarse-to-fine neural deformation model to simultaneously recover the camera pose and the 4D reconstruction of an unknown object from multiple RGB sequences in the wild. To that end, our approach does not consider any pre-built 3D template nor 3D training data as well as controlled illumination conditions, and can sort out the problem in a self-supervised manner. Our model exploits canonical and image-variant spaces where both coarse and fine components are considered. We introduce a neural local quadratic model with spatio-temporal consistency to encode fine details that is combined with canonical embeddings in order to establish correspondences across sequences. We thoroughly validate the method on challenging scenarios with complex and real-world deformations, providing both quantitative and qualitative evaluations, an ablation study and a comparison with respect to competing approaches.

![teaser](https://github.com/user-attachments/assets/21e48565-2caf-4e96-a61e-1af357a4aacf)

# Install instructions:

For an easy setup that creates a conda environment 4DPV and downloads all the baseline datasets:
```
git clone https://github.com/smontode24/4DPV.git --recursive
cd 4DPV
./initial_setup.sh
```

### Data
**To use your own videos, or pre-process raw videos into banmo format, 
please follow the instructions [here](./preprocess).**


#### Optimization 

cat-pikachiu sequence:
```
seqname=cat-pikachiu
# To speed up data loading, we store images as lines of pixels). 
# only needs to run it once per sequence and data are stored
python preprocess/img2lines.py --seqname $seqname

# Optimization
bash scripts/run_experiment_longer_steps.sh 0,1 cat-pikachiu 8900 "no" "no" 26 training_ablation_FIXFINE_quadratic_noIPE 25 --photometric_loss l1 --num_iters_small_training 400 --eval_every_n 20 --nerf_n_layers_c 8 --nerf_hidden_dim_c 256 --nerf_n_layers_f 8 --nerf_hidden_dim_f 256 --num_freqs 10 --fine_nerf_net --extra_dfm_nr --dfm_type quadratic --sil_wt 1 --img_wt 0.1 --wr 1e3 --wsd 1e-2 --nouse_window --nolarge_cond_embedding --nofine_samples_from_coarse --noone_cycle_lr --reset_lr --bound_fine --use_sdf_finenerf --use_separate_code_dfm --combine_dfm_and_pose --dfm_emb_dim 64 --render_size 32 --smoothness_dfm --smoothness_dfm_spat --train_always_unc --nsample_a_mult 16 --mult_img_wt_on_refine --img_wt_mlt 10 --weight_smoothness_opacity --neighborhood_scale 1e-2
# argv[1]: gpu ids separated by comma 
# args[2]: sequence name
# args[3]: port for distributed training
# args[4]: use_human, pass "" for human cse, "no" for quadreped cse
# args[5]: use_symm, pass "" to force x-symmetric shape
# args[6]: batch size
# args[7]: log name
# args[8]: number of bones

# Extract articulated meshes and render
bash scripts/render_vids.sh cat-pikachiu logdir/cat-pikachiu-e120-b26-training_ablation_FIXFINE_quadratic_noIPE-b25-retrained-ft2/params_latest.pth "0" --nosymm_shape --nouse_human --num_bones 25 --photometric_loss l1 --num_iters_small_training 400 --eval_every_n 20 --nerf_n_layers_c 8 --nerf_hidden_dim_c 256 --nerf_n_layers_f 8 --nerf_hidden_dim_f 256 --num_freqs 10 --fine_nerf_net --extra_dfm_nr --dfm_type quadratic --sil_wt 1 --img_wt 0.1 --wr 1e3 --wsd 1e-2 --nouse_window --nolarge_cond_embedding --nofine_samples_from_coarse --noone_cycle_lr --reset_lr --bound_fine --use_sdf_finenerf --use_separate_code_dfm --combine_dfm_and_pose --dfm_emb_dim 64 --render_size 32 --smoothness_dfm --smoothness_dfm_spat --train_always_unc --nsample_a_mult 16 --mult_img_wt_on_refine --img_wt_mlt 10 --weight_smoothness_opacity --neighborhood_scale 1e-2
# argv[1]: sequence name 
# args[2]: pretrained weights
# args[3]: reference video id
```

ama-female sequence:
```
bash scripts/run_experiment_longer_steps.sh 0,1,2,3 ama-female 10002 "" "no" 26 training_ama_female 25 --photometric_loss l1 --num_iters_small_training 400 --eval_every_n 20 --nerf_n_layers_c 8 --nerf_hidden_dim_c 256 --nerf_n_layers_f 8 --nerf_hidden_dim_f 256 --num_freqs 10 --fine_nerf_net --extra_dfm_nr --dfm_type quadratic --sil_wt 10 --img_wt 0.1 --wr 1e4 --wsd 1e-2 --nouse_window --nolarge_cond_embedding --nofine_samples_from_coarse --noone_cycle_lr --reset_lr --use_sdf_finenerf --use_separate_code_dfm --combine_dfm_and_pose --dfm_emb_dim 64 --render_size 32 --smoothness_dfm --smoothness_dfm_spat --train_always_unc --nsample_a_mult 16 --mult_img_wt_on_refine --img_wt_mlt 10 --weight_smoothness_opacity --neighborhood_scale 1e-3
```

a-eagle sequence:
```
bash scripts/run_experiment_longer_steps.sh 0 a-eagle 10002 "no" "no" 84 training_eagle 25 --photometric_loss l1 --num_iters_small_training 400 --eval_every_n 20 --nerf_n_layers_c 8 --nerf_hidden_dim_c 256 --nerf_n_layers_f 8 --nerf_hidden_dim_f 256 --num_freqs 10 --fine_nerf_net --extra_dfm_nr --dfm_type quadratic --sil_wt 10 --img_wt 0.1 --wr 1e5 --wsd 1e-2 --nouse_window --nolarge_cond_embedding --nofine_samples_from_coarse --noone_cycle_lr --reset_lr --bound_fine --use_sdf_finenerf --use_separate_code_dfm --combine_dfm_and_pose --dfm_emb_dim 64 --smoothness_dfm --smoothness_dfm_spat --train_always_unc --nsample_a_mult 16 --mult_img_wt_on_refine --img_wt_mlt 10 --weight_smoothness_opacity --neighborhood_scale 1e-4
```


#### 2. Visualization tools
Tensorboard:
```
# You may need to set up ssh tunneling to view the tensorboard monitor locally.
screen -dmS "tensorboard" bash -c "tensorboard --logdir=logdir --bind_all"
```

#### Repo Status
- [x] Initial code release
- [ ] Upload dog sequence
- [ ] Provide more examples

### License
<details><summary>[expand]</summary>

- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
</details>
