 #!/bin/bash

#PBS -S /bin/bash
#PBS -N auto-enc-nico
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=2:gputype=k40m
#PBS -q gpuq
#PBS -P aud_interp
#PBS -M nicolas.gabrion@student.ecp.fr
#PBS -o logs/output_autoencLstm.txt
#PBS -e logs/error_autoencLstm.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/2017gabrionn/AudioMNIST

source activate aud_interp_gpu

mprof run -o "logs/mprofile_nico.dat" train_autoenclstm.py -i tf_data/audionet.tfrecords -o models/autoenclstm -l tensorboard/autoenclstm1 -b 100 -e 10
