 #!/bin/bash

#PBS -S /bin/bash
#PBS -N autoencConv-test-dty
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=2:gputype=k40m
#PBS -q gpuq
#PBS -P randstad
#PBS -M badryoubiidrissi@gmail.com
#PBS -o logs/output_autoencConv3.txt
#PBS -e logs/error_autoencConv3.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/idrissib/AudioMNIST

source activate aud_interp_gpu

mprof run -o "logs/mprofile_<autoencConv>.dat" train_wavenet.py -i tf_data/audionet.tfrecords -o models/autoencConv3 -l tensorboard/autoencConv3 -b 100 -e 50