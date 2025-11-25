# AgentSimulator-Optimizer
This is the supplementary GitHub repository of the paper: "From Global Policies to Local Strategies: Multi-Objective Optimization of Resource-Specific Handover Policies".

### Abstract
Efficient resource allocation is a key challenge in business process management, with direct implications for cost, throughput time, and utilization. While recent Reinforcement Learning (RL) approaches have shown promise in deriving adaptive allocation policies, they typically neglect inter-resource collaboration patterns that can strongly influence real-world task handovers. This paper introduces the first approach for multi-objective optimization of resource-level decision-making, enabling the discovery of person-specific handover policies. Building on a Multi-Agent System (MAS)-based process simulator, we extend it from a descriptive to a prescriptive optimization environment by integrating the Non-dominated Sorting Genetic Algorithm-II (NSGA-II). The resulting approach learns Pareto-optimal, resource-specific policies that optimize the process across multiple objectives. Experimental results on synthetic and real-world datasets show that our approach reduces cost by an average of 37% and waiting time by 58%, consistently outperforming heuristic baselines and demonstrating the potential of leveraging collaboration-aware optimization for process performance improvement.

### Environment setup (Conda + pip)

The project uses Python packages pinned in `requirements.txt`. The recommended setup is a fresh Conda environment and installing packages via pip inside that env.

```bash
# 1) Create and activate a new environment 
conda create -n agent_simulator python=3.9 
conda activate agent_simulator

# 2) Optional: ensure pip is available/updated in the env
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt
```

### How to run the code
The main script is `optimize_pro_plus.py`.
You need to specify the following parameters regarding the input event log:

`--log_path`: path to the entire log which you need to store in the folder raw_data

`--case_id`: name of the case_id column

`--activity_name`: name of the activity column

`--resource_name`: name of the resource column

`--end_timestamp`: name of the end timestamp column

`--start_timestamp`: name of the start timestamp column

`--costs_path`: if cost is an objective, you need to specify the path to the json file that assigns hourly costs per resource

Then, you can specify the following parameters for the optimization:

`--pop_size`: size of the population

`--num_gen`: number of generations for the GA

`--runs_per_fitness`: number of simulations per policy evaluation

`--mut_prob`: mutation probability

`--cx_prob`: crossover probability

`--objectives`: objectives to optimize (e.g., "cost,wait")

`--mutation_choice`: mutation variant (0.5 for Guided, 0.7 for Greedy, 1 for Random, empty for Hybrid)

All commands to run the datasets evaluated in our paper are in jobs/caise/.

We provide additional results in supplementary_material.pdf.

## Authors
Lukas Kirchdorfer, Artemis Doumeni, Han van der Aa, Hugo A. López
