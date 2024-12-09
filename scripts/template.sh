# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
runname=$7
num_bones=25 # Try other amount
num_epochs=120
batch_size=$6 # 256 by default

model_prefix=$seqname-e$num_epochs-b$batch_size-$runname-b$num_bones-retrained
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

# The multi-stage optimization will be described in the camera-ready version. 
# In short, the result of the first stage is typically good enough. Stage 2-3
# improve the limb articulations (e.g., to correctly articulate the crossing legs 
# for cat-pikachiu) with coordinate descent. Stage 4 further improves details of 
# the geometry. Bones are not re-initialized in stage 2-4.

##############################################
##### STEP 1 #################################
##############################################
# mode: line load
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --${use_symm}symm_shape \
  --${use_human}use_human --num_bones $num_bones \
  --photometric_loss huber

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
loadname=${model_prefix}-init
savename=${model_prefix}-ft1
num_epochs=$((num_epochs/4)) 

##############################################
##### STEP 2 #################################
##############################################
# Extra things compared to first:
# - model_path -> Load last parameters
# - warmup_steps -> 0? No warmup silhouette loss (not used?) , steps used to increase sil loss
# - nf_reset: reset near-far plane at 100% (default 50%)
# - bound_reset: reset hyper-parameters based on current geometry / cameras at 100% (default 50%)
# - dskin_steps: how much percentage to wait until adding delta skinning weights: 0% (use from the start)
# - fine_steps: When to use fine_steps (fine NeRF not used)
# - noanneal_freq: 
# - freeze_proj: 0-80% body pose with proj loss, 80-100% gradually add all loss (freeze shape/feature etc) -> Not sure, see next param
# - proj_end: steps to end projection opt set to 1, then projection loss tuning is never used?
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1\
  --${use_symm}symm_shape \
  --${use_human}use_human --num_bones $num_bones \
  --photometric_loss huber

##############################################
##### STEP 3 #################################
##############################################
# mode: fine tune with active+fine samples, large rgb loss wt and reset beta (USING FINE NERF)
# Differences: 
# - Using always fine NeRF
# - reset near far plane at the start, reset hyper-parameters at start
# - use uncertainty MLP
# - img_wt=1 (increase silhouette loss)
# - reset_beta: reset volsdf beta to 0.1 (what is volsdf ? Used in coarse NeRF)
loadname=${model_prefix}-ft1
savename=${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --${use_symm}symm_shape \
  --${use_human}use_human --num_bones $num_bones \
  --photometric_loss huber

##############################################
##### STEP 4 #################################
##############################################
# Don't use optical flow, CSE-related losses and others that limit the synthesis performance
# - Disable flow loss: --nouse_corresp
# - Disable projected CSE l2 loss: --nouse_cseproj
# - Disable similarity of 3D point to canonical mesh: --nouse_embed
# - ...
# Change laplacian for density? If it does not produce good enough results

loadname=${model_prefix}-ft2
savename=${model_prefix}-ft3
#num_epochs=$((num_epochs*4))
num_epochs=20
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --${use_symm}symm_shape \
  --${use_human}use_human --num_bones $num_bones \
  --nouse_corresp --nouse_cseproj --noft_cse --nomt_cse \
  --nouse_embed --nouse_proj --nosmall_training --photometric_loss huber
