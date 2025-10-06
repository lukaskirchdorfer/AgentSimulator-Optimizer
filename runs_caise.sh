#!/bin/bash
#SBATCH --job-name=LoanApp_base
#SBATCH --cpus-per-task=20
#SBATCH --mem=70G
#SBATCH --partition=cpu
#SBATCH --chdir=/ceph/lkirchdo/agent_optimizer/AgentSimulator-Optimizer

# Ensure conda is available
source ~/.bashrc || { echo "Failed to source ~/.bashrc"; exit 1; }

# Activate conda environment
conda activate agent-opt || { echo "Failed to activate agent-opt environment"; exit 1; }

# Navigate to project directory
echo "Changing directory to the agent-opt repository..."
cd /ceph/lkirchdo/agent_optimizer/AgentSimulator-Optimizer || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# -------------------------------
echo "Starting Python script execution..."
echo "-------------------------------------"

# Random
python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
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

# # Mix
# python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
#   --case_id case_id \
#   --activity_name activity \
#   --resource_name resource \
#   --start_timestamp start_time \
#   --end_timestamp end_time \
#   --pop_size 100 \
#   --num_gen 100 \
#   --runs_per_fitness 100 \
#   --n_cores 20 \
#   --mut_prob 0.3 \
#   --cx_prob 0.7 \
#   --costs_path costs.json \
#   --objectives "cost,wait" \

# # Guided
# python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
#   --case_id case_id \
#   --activity_name activity \
#   --resource_name resource \
#   --start_timestamp start_time \
#   --end_timestamp end_time \
#   --pop_size 100 \
#   --num_gen 100 \
#   --runs_per_fitness 100 \
#   --n_cores 20 \
#   --mut_prob 0.3 \
#   --cx_prob 0.7 \
#   --costs_path costs.json \
#   --objectives "cost,wait" \
#   --mutation_choice 0.5

#   # Greedy
#   python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
#   --case_id case_id \
#   --activity_name activity \
#   --resource_name resource \
#   --start_timestamp start_time \
#   --end_timestamp end_time \
#   --pop_size 100 \
#   --num_gen 100 \
#   --runs_per_fitness 100 \
#   --n_cores 20 \
#   --mut_prob 0.3 \
#   --cx_prob 0.7 \
#   --costs_path costs.json \
#   --objectives "cost,wait" \
#   --mutation_choice 0.7