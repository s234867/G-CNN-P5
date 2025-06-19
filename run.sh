#!/bin/bash
### LSF‑options ###
#BSUB -q gpuv100
#BSUB -J my_gpu_job
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[sxm2]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -u din_email@domæne.dk
#BSUB -Ne
#BSUB -o my_gpu_job.%J.out
#BSUB -e my_gpu_job.%J.err
#BSUB -cwd /zhome/d1/3/206707/Desktop/G-CNN-P5
#BSUB -W 24:00 
### end of options ###

# Purge any old modules and load the ones we need
module purge
module load python3/3.10.16 cuda/11.6

# Optional: confirm GPU availability under this module
nvidia-smi

# Activate the virtual environment
source G-CNN-env/bin/activate

# Run your training
python src/main.py
