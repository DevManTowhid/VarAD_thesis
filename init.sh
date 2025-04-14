## CUDA: 12.2
## NVCC: 11.8

conda create -n varad_env python=3.9.5 -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# there can be some remained bugs in installing the environment...
# for nvcc -V 11.8
#conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc
#wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#chmod +x https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
##and pay attention to the paths in this step. option/paths for only the paths that you have admission
#sh https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#export CUDA_HOME=$CUDA_HOME:/home/anyad/cuda-11.8
#export PATH=$PATH:/home/anyad/cuda-11.8/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anyad/cuda-11.8/lib64

pip install tqdm tensorboard setuptools==58.0.4 opencv-python scikit-image scikit-learn matplotlib seaborn ftfy regex

cd ./VMamba
pip install -r requirements.txt

cd kernels/selective_scan && pip install .

pip3 install imgaug
pip3 install Cython

### for dataset: remember to change the config/global config for the root dir!
