#sudo apt-get update; sudo apt-get install build-essential git; sudo apt-get install cmake; sudo apt-get update
mkdir tmp; mkdir logdir
git submodule sync
git submodule update --init --recursive
#conda env create -f environment.yml
conda create --yes -n 4DPV python=3.9
conda activate 4DPV
conda install --yes pip
conda install --yes pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# Prerequisites pytorch3d
conda install --yes -c fvcore -c iopath -c conda-forge fvcore iopath
conda install --yes -c bottler nvidiacub
# install pytorch3d (takes minutes), kmeans-pytorch
#python -m pip install -e third_party/pytorch3d # or install with: 
conda install --yes pytorch3d -c pytorch3d
python -m pip install -e third_party/kmeans_pytorch
# install detectron2
# you can also try: 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html # Must be adapted if using different torch version
# install tinycudann
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# install requirements
python -m pip install -r requirements.txt
# download preprocessed videos
bash misc/processed/download.sh cat-pikachiu
bash misc/processed/download.sh human-cap
# download pretrained posenet weights
mkdir -p mesh_material/posenet && cd "$_"
wget $(cat ../../misc/posenet.txt); cd ../../
# download preoptimized models
mkdir -p tmp && cd "$_"
wget https://www.dropbox.com/s/qzwuqxp0mzdot6c/cat-pikachiu.npy
wget https://www.dropbox.com/s/dnob0r8zzjbn28a/cat-pikachiu.pth
wget https://www.dropbox.com/s/p74aaeusprbve1z/opts.log # flags used at opt time
cd ../
mkdir logdir/cat-pickachiu-pretrained; mv tmp/cat-pikachiu.pth logdir/cat-pickachiu-pretrained; mv tmp/cat-pikachiu.npy logdir/cat-pickachiu-pretrained; mv tmp/opts.log logdir/cat-pickachiu-pretrained
python preprocess/img2lines.py --seqname cat-pikachiu
git lfs install; git lfs pull
# sudo apt-get install git-lfs; git lfs install; git lfs pull
