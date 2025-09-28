#!/bin/sh
#BSUB -J o_BPI_2017W_mini
#BSUB -o out/o_BPI_2017W_mini_%J.out
#BSUB -e err/o_BPI_2017W_mini_%J.err
#BSUB -n 20
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 20:00
#BSUB -R 'rusage[mem=32GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python -u optimize_pro_plus.py --log_path raw_data/BPI_2017W.csv \
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
  --costs_path costs_BPI_2017W.json \
  --objectives "cost,wait"