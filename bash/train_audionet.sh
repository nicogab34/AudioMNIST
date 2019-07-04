 #!/bin/bash

#PBS -S /bin/bash
#PBS -N ELMo-test
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=2:gputype=p100
#PBS -q gpuq
#PBS -P randstad
#PBS -M badryoubiidrissi@gmail.com
#PBS -o logs/output.txt
#PBS -e logs/error.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/idrissib/AudioMNIST

source activate aud_interp_gpu

python train_classification.py audionet_big -i tf_data/raw.tfrecords -o models/audionet_big_1 -l tensorboard/audionet_big_1 -b 100 -e 50 -lr 0.0005