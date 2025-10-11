#!/bin/bash

# Navigate to project directory
echo "Changing directory to the agent-opt repository..."
cd /Users/I589354/Library/CloudStorage/OneDrive-SAPSE/Development/AgentSimulator-Optimizer || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

echo "=========================================="
echo "Running all dataset optimization experiments"
echo "=========================================="

# Function to run a script and handle errors
run_script() {
    local script_path=$1
    local dataset_name=$2
    
    echo ""
    echo "----------------------------------------"
    echo "Starting: $dataset_name"
    echo "Script: $script_path"
    echo "----------------------------------------"
    
    if [ -f "$script_path" ]; then
        echo "Executing $script_path..."
        bash "$script_path"
        
        if [ $? -eq 0 ]; then
            echo "✅ $dataset_name completed successfully"
        else
            echo "❌ $dataset_name failed with exit code $?"
        fi
    else
        echo "❌ Script not found: $script_path"
    fi
    
    echo "----------------------------------------"
}

# Run all dataset experiments
echo "Starting dataset experiments..."

# ACR dataset
run_script "jobs/caise/ACR/all.sh" "ACR"

# BPI12W dataset  
run_script "jobs/caise/BPI12W/all.sh" "BPI12W"

# Confidential_1000 dataset
run_script "jobs/caise/Confidential_1000/all.sh" "Confidential_1000"

# Confidential_2000 dataset
run_script "jobs/caise/Confidential_2000/all.sh" "Confidential_2000"

# cvs_pharmacy dataset
run_script "jobs/caise/cvs_pharmacy/all.sh" "cvs_pharmacy"

# BPI17W dataset
run_script "jobs/caise/BPI17W/all.sh" "BPI17W"

# # P2P dataset
# run_script "jobs/caise/P2P/all.sh" "P2P"

# # LoanApp base dataset
# run_script "jobs/caise/LoanApp/base/all.sh" "LoanApp_base"

# # LoanApp activity_dependent dataset
# run_script "jobs/caise/LoanApp/activity_dependent/all.sh" "LoanApp_activity_dependent"

# # LoanApp junior_senior dataset
# run_script "jobs/caise/LoanApp/junior_senior/all.sh" "LoanApp_junior_senior"

# # LoanApp same_money_diff_time dataset
# run_script "jobs/caise/LoanApp/same_money_diff_time/all.sh" "LoanApp_same_money_diff_time"

echo ""
echo "=========================================="
echo "All dataset experiments completed!"
echo "=========================================="
