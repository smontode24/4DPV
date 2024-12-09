# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
export MKL_SERVICE_FORCE_INTEL=1
dev=$1
ngpu=`echo $dev |  awk -F '[\t,]' '{print NF-1}'`
ngpu=$(($ngpu + 1 ))
echo "using "$ngpu "gpus"

logname=$2
seqname=$3
address=$4
add_args=${*: 4:$#-1}

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch\
                    --master_addr="127.0.0.1" \
                    --master_port=$address \
                    --nproc_per_node=$ngpu --use_env main.py \
                    --ngpu $ngpu \
                    --seqname $seqname \
                    --logname $logname \
                    $add_args
# -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=8999 --use_env