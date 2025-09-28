#!/bin/sh
#BSUB -J o_random_P2P
#BSUB -o out/o_random_P2P%J.out
#BSUB -e err/o_random_P2P%J.err
#BSUB -n 20
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 20:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python -u optimize_pro_plus.py --log_path raw_data/P2P.csv \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 100 \
  --num_gen 100 \
  --runs_per_fitness 100 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs.json \
  --objectives "cost,wait" \
  --mutation_choice 1