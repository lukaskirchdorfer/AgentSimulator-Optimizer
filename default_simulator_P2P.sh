#!/bin/sh
#BSUB -J simple_simulator_P2P
#BSUB -o out/simple_simulator_P2P_%J.out
#BSUB -e err/simple_simulator_P2P_%J.err
#BSUB -n 1
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 2:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python simulate.py --log_path raw_data/P2P.csv \
       --case_id case:concept:name \
       --activity_name concept:name \
       --resource_name Resource \
       --end_timestamp time:timestamp \
       --start_timestamp start_timestamp 
