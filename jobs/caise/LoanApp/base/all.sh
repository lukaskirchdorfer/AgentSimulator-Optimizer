# Navigate to project directory
echo "Changing directory to the agent-opt repository..."
cd /Users/I589354/Library/CloudStorage/OneDrive-SAPSE/Development/AgentSimulator-Optimizer || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# -------------------------------
echo "Starting Python script execution..."
echo "-------------------------------------"

# Greedy
python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 100 \
  --num_gen 100 \
  --runs_per_fitness 3 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs/costs.json \
  --objectives "cost,wait" \
  --mutation_choice 0.7

# Guided
python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 100 \
  --num_gen 100 \
  --runs_per_fitness 3 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs/costs.json \
  --objectives "cost,wait" \
  --mutation_choice 0.5

# Mixed
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
  --costs_path costs/costs.json \
  --objectives "cost,wait" \

# Random
python -u optimize_pro_plus.py --log_path raw_data/LoanApp.csv.gz \
  --case_id case_id \
  --activity_name activity \
  --resource_name resource \
  --start_timestamp start_time \
  --end_timestamp end_time \
  --pop_size 100 \
  --num_gen 100 \
  --runs_per_fitness 3 \
  --n_cores 20 \
  --mut_prob 0.3 \
  --cx_prob 0.7 \
  --costs_path costs/costs.json \
  --objectives "cost,wait" \
  --mutation_choice 1