#!/bin/sh
#BSUB -J simple_simulator
#BSUB -o out/simple_simulator_%J.out
#BSUB -e err/simple_simulator_%J.err
#BSUB -n 1
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 2:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python simulate.py --log_path raw_data/BPI2012P.csv \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --end_timestamp end_time \
  --start_timestamp start_time