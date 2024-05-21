#!/bin/bash
#BSUB -J mas-solu-1l
#BSUB -o logs/mas-solu-1l_%J.out
#BSUB -e logs/mas-solu-1l_%J.err
#BSUB -q gpua100
#BSUB -n 12
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096]"
#BSUB -gpu "num=1:mode=exclusive_process"

module load python3/3.11.3
module load nvhpc/22.11-nompi
module load cuda/11.8

source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m thesis.mas --config-name gelu-1l-sae.yaml params.samples_to_check=2097152

