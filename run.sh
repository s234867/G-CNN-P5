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
#BSUB -cwd /zhome/d1/3/206707/Desktop/Fagprojekt
### end of options ###

# Vis GPU‑status
nvidia-smi

# Ryd moduler & load kun CUDA
module purge
module load cuda/11.6

# Aktivér dit venv korrekt
source test-env/bin/activate

# Kør din kode
python src/main.py

