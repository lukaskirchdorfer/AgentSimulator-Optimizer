#!/bin/sh
#BSUB -J o_random_ACR_mini
#BSUB -o out/o_random_ACR_mini%J.out
#BSUB -e err/o_random_ACR_mini%J.err
#BSUB -n 20
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 20:00
#BSUB -R 'rusage[mem=8GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python -u optimize_pro_plus.py --log_path raw_data/ACR.csv \
  --case_id case_id \
  --activity_name activity_name \
  --resource_name resource \
  --start_timestamp start_timestamp \
  --end_timestamp end_timestamp \
  --extr_delays \
  --pop_size 20 \
  --num_gen 10 \
  --runs_per_fitness 100 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs_ACR.json \
  --objectives "cost,wait" \
  --mutation_choice 1