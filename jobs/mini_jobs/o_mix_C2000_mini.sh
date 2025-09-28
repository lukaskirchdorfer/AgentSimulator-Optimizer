#!/bin/sh
#BSUB -J o_mix_C2000_mini
#BSUB -o out/o_mix_C2000_mini%J.out
#BSUB -e err/o_C200o_mix_C2000_mini0_mini_%J.err
#BSUB -n 20
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 20:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python -u optimize_pro_plus.py --log_path raw_data/C2000.csv \
  --case_id case_id \
  --activity_name activity_name \
  --resource_name resource \
  --start_timestamp start_timestamp \
  --end_timestamp end_timestamp \
  --pop_size 20 \
  --num_gen 10 \
  --runs_per_fitness 100 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs_C2000.json \
  --objectives "cost,wait"