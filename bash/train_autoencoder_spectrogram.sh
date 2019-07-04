 #!/bin/bash

#PBS -S /bin/bash
#PBS -N ELMo-test
#PBS -l walltime=2:00:00
#PBS -l select=2:ncpus=2:gputype=k40m
#PBS -q gpuq
#PBS -P aud_interp
#PBS -M julien.duquesne@student.ecp.fr
#PBS -o logs/output.txt
#PBS -e logs/error.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/2017duquesnej/AudioMNIST

source activate aud_interp_gpu

mprof run -o 'logs/mprofile_<YYYYMMDDhhmmss>.dat' train_autoencoder_spectrogram.py -i tf_data/spectrogram.tfrecords -o models/autoencoder_spectrogram -l tensorboard/autoencoder_spectrogram_lr_0.0005 -b 100 -e 50