#!/bin/sh
#BSUB -J ga_plus_wt
#BSUB -o out/ga_plus_wt%J.out
#BSUB -e err/ga_plus_wt%J.err
#BSUB -n 10
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 8:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -u s234061@student.dtu.dk
#BSUB -N

source agent/bin/activate

python optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 20 \
  --num_gen 10 \
  --runs_per_fitness 100 \
  --n_cores 20 \
  --costs_path costs.json \
  --objectives "cost,wait,agents_per_case"