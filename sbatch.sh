#!/bin/bash
#SBATCH --gres=gpu:1     # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G      # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:20
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=mrsunchen0110@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip
cd spikingjelly
python setup.py install
pip list
cd ..
cd c10
module load cuda
module load cudnn
pip install --no-index torch==1.9.1 torchvision==0.9.1 
pip install thop
pip install  tensorboard==2.2.1 --no-index
pip list

nvidia-smi                        # you can use 'nvidia-smi' for a test
# ResNet56 on CIFAR-10

python c10.py -s 0.95  -gpu 0 --dataset-dir ../data_cifar10 --dump-dir dump -m grad
wait
echo "Done"
