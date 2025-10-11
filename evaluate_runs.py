# evaluate_runs.py (Version 7 - Dynamic Baselines)
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# --- Parsing and Pareto Front Logic ---
# =============================================================================

def parse_run_summary(file_path):
    """Parses a run_summary.txt file to extract key information with dynamic baselines."""
    data = {'variant_name': '', 'baselines': {}, 'pareto_front': []}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Heuristic for naming variants based on file path, can be customized
    if 'config_0.5' in file_path: data['variant_name'] = 'Guided'
    elif 'config_0.7' in file_path: data['variant_name'] = 'Greedy'
    elif 'config_1.0' in file_path: data['variant_name'] = 'Random Scramble'
    elif 'config_mix' in file_path: data['variant_name'] = 'Mixed Strategy'
    else: data['variant_name'] = os.path.basename(os.path.dirname(file_path))

    in_baseline_section = False
    expect_baseline_detail_for = None

    in_pareto_section = False

    for i, raw in enumerate(lines):
        line = raw.strip()

        # Detect sections
        if line.startswith('--- Baseline Performance'):
            in_baseline_section = True
            expect_baseline_detail_for = None
            continue
        if line.startswith('--- Optimization Results') or line.startswith('--- Pareto Front Solutions'):
            in_baseline_section = False
            expect_baseline_detail_for = None

        # Parse dynamic baselines
        if in_baseline_section:
            m = re.match(r'^Simulated Baseline \(([^)]+)\):', line)
            if m:
                expect_baseline_detail_for = m.group(1)
                # The next non-empty line should contain the details with '->'
                continue

            if expect_baseline_detail_for:
                # Try to parse details line
                # Example: "-> Cost: $85,549.21136128, Time: 14.41656583h, Wait: 0.66065359h, ..."
                cost_match = re.search(r"Cost:\s*\$([\d,]*\d(?:\.\d+)?)", line)
                wait_match = re.search(r"Wait:\s*([\d\.]+)h", line)
                if cost_match and wait_match:
                    cost_val = float(cost_match.group(1).replace(',', ''))
                    wait_val = float(wait_match.group(1))
                    data['baselines'][expect_baseline_detail_for] = {'cost': cost_val, 'wait': wait_val}
                    expect_baseline_detail_for = None
                # keep scanning until we find the details or a new header/section
                continue

        # This regex is specific to your optimizer's output format for the pareto front
        if re.match(r"Sol\s*\|\s*Cost\s*\|", line):
            in_pareto_section = True
            continue

        if in_pareto_section:
            if re.match(r"^-+\|-+", line):
                continue
            if '---' in line or not line:
                in_pareto_section = False
                continue

            # Extracts the numbers from the cost and wait columns
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[1] and parts[2]:
                try:
                    cost = float(parts[1])
                    wait = float(parts[2])
                    data['pareto_front'].append((cost, wait))
                except (ValueError, IndexError):
                    pass
    
    return data

def is_dominated(p1, p2):
    """Check if point p1 is dominated by point p2 (for minimization)."""
    return all(p2[i] <= p1[i] for i in range(len(p1))) and any(p2[i] < p1[i] for i in range(len(p1)))

def get_pareto_front(points):
    """Filters a list of points to return only the non-dominated ones."""
    if not points:
        return np.array([])
    points_np = np.array(points)
    is_efficient = np.ones(points_np.shape[0], dtype=bool)
    for i, p1 in enumerate(points_np):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points_np[is_efficient] < p1, axis=1) | np.all(points_np[is_efficient] == p1, axis=1)
            is_efficient[i] = True 
    return points_np[is_efficient]

# =============================================================================
# --- CORRECTED & NEW METRIC CALCULATION FUNCTIONS ---
# =============================================================================

def calculate_hyperarea(points, ref_point):
    """Calculates the 2D hyperarea for minimization problems."""
    if len(points) == 0:
        return 0.0
    
    sorted_points = sorted([tuple(p) for p in points], key=lambda p: p[0])
    
    area = 0.0
    last_y = ref_point[1]
    
    for x, y in sorted_points:
        width = ref_point[0] - x
        height = last_y - y
        if width > 0 and height > 0:
            area += width * height
        last_y = y
        
    return area

def calculate_averaged_hausdorff(front_approx, front_ref):
    """Calculates the Averaged Hausdorff Distance."""
    if front_approx.size == 0 or front_ref.size == 0:
        return float('inf')

    def d_avg(A, B):
        return np.mean([np.min(np.sqrt(np.sum((B - p_a)**2, axis=1))) for p_a in A])

    return max(d_avg(front_approx, front_ref), d_avg(front_ref, front_approx))

def calculate_purity(front_approx, front_ref):
    """Calculates purity."""
    if len(front_approx) == 0: return 0.0
    ref_set = {tuple(p) for p in front_ref}
    matches = sum(1 for p in front_approx if tuple(p) in ref_set)
    return matches / len(front_approx)

def calculate_delta_spread(front_approx, front_ref):
    """Calculates the Delta spread metric for 2D fronts."""
    if front_approx.size < 2 or front_ref.size < 2:
        return float('inf')

    yn = front_approx[front_approx[:, 0].argsort()]
    pref_sorted = front_ref[front_ref[:, 0].argsort()]
    
    yn_extreme = np.array([yn[0], yn[-1]])
    pref_extreme = np.array([pref_sorted[0], pref_sorted[-1]])
    
    d_first = np.linalg.norm(yn_extreme[0] - pref_extreme[0])
    d_last = np.linalg.norm(yn_extreme[1] - pref_extreme[1])
    
    distances = np.linalg.norm(yn[1:] - yn[:-1], axis=1)
    d_avg = np.mean(distances)
    
    numerator = d_first + d_last + np.sum(np.abs(distances - d_avg))
    denominator = d_first + d_last + (len(yn) - 1) * d_avg
    
    return numerator / denominator if denominator > 0 else float('inf')

# =============================================================================
# --- Most Balanced Solution Logic ---
# =============================================================================
def find_most_balanced_solution(points_np):
    """
    Finds the most balanced solution from a Pareto front.
    This replicates the logic from optimizer.py by finding the point
    closest to the origin in a normalized objective space.
    """
    print(f"points_np: {points_np}")
    if points_np.size == 0:
        return None
    if len(points_np) == 1:
        return points_np[0]

    min_vals = points_np.min(axis=0)
    ptp_values = points_np.ptp(axis=0)
    
    denominator = np.where(ptp_values == 0, 1, ptp_values)
    norm_points = (points_np - min_vals) / denominator
    
    distances = np.linalg.norm(norm_points, axis=1)
    
    best_index = np.argmin(distances)

    print(f"best solution: {points_np[best_index]}")
    
    return points_np[best_index]

# =============================================================================
# --- Main Evaluation Logic ---
# =============================================================================
def main():
    print("--- Starting Evaluation of Optimization Runs ---")
    
    # --- Part 1: Data Aggregation and Preparation ---
    DATASET = 'LoanApp'

    search_dir = f'optimization_runs/{DATASET}'
    
    all_run_data = []
    for dirpath, _, filenames in os.walk(search_dir):
        if 'run_summary.txt' in filenames:
            file_path = os.path.join(dirpath, 'run_summary.txt')
            parsed_data = parse_run_summary(file_path)
            print(f"Parsed data: {parsed_data}")
            if parsed_data['pareto_front']:
                all_run_data.append(parsed_data)

    if not all_run_data:
        print(f"Error: No valid 'run_summary.txt' files found with solutions. Exiting.")
        return

    print(f"Found {len(all_run_data)} valid run summaries to evaluate.")
    all_run_data.sort(key=lambda x: x['variant_name'])

    all_solutions = [sol for data in all_run_data for sol in data['pareto_front']]
    unique_solutions = list(set(map(tuple, all_solutions)))
    print(f"Total unique solutions found across all runs: {len(unique_solutions)}")

    # --- Part 2: High-Level Improvement Analysis ---
    # Now prints improvements vs. each dynamic baseline present in the summary files.
    print("\n--- High-Level Improvement Analysis (Mean of Front vs. Baselines) ---\n")
    for data in all_run_data:
        front_np = np.array(data['pareto_front'])
        mean_cost, mean_wait = front_np.mean(axis=0)
        print(f"Variant: {data['variant_name']}")
        if not data['baselines']:
            print("  (No baselines found)")
        else:
            for bname, bvals in data['baselines'].items():
                cost_improv = (bvals['cost'] - mean_cost) / bvals['cost'] * 100 if bvals.get('cost') else 0.0
                wait_improv = (bvals['wait'] - mean_wait) / bvals['wait'] * 100 if bvals.get('wait') else 0.0
                print(f"  - Baseline '{bname}': Mean Cost Impr: {cost_improv:.2f}% | Mean Wait Impr: {wait_improv:.2f}%")
        print("")

    # --- Part 3: Rigorous Front-Quality Analysis ---
    print("\n--- Detailed Front Quality Analysis (vs. Reference Front) ---")
    pref = get_pareto_front(unique_solutions)
    print(f"Constructed a Reference Pareto Front with {len(pref)} solutions.")
    if pref.size == 0:
        print("Reference front is empty. Cannot perform quality analysis.")
    else:
        max_cost = max(p[0] for p in unique_solutions) * 1.01
        max_wait = max(p[1] for p in unique_solutions) * 1.01
        ref_point = (max_cost, max_wait)
        print(f"Using reference point for Hyperarea: Cost={ref_point[0]:.2f}, Wait={ref_point[1]:.2f}")

        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c'] 
        markers = ['s', 'o', '^', 'D']
        
        for i, data in enumerate(all_run_data):
            front_np = np.array(data['pareto_front'])
            plt.scatter(front_np[:, 0], front_np[:, 1], 
                        c=colors[i % len(colors)], marker=markers[i % len(markers)], s=50, 
                        label=f"{data['variant_name']} ({len(front_np)} sols)")

        plt.scatter(pref[:, 0], pref[:, 1], c='black', marker='x', s=100, alpha=0.7,
                    label=f'Reference Front (PRef) ({len(pref)} sols)', zorder=10)
        plt.scatter(ref_point[0], ref_point[1], c='purple', marker='X', s=100,
                    label='Hyperarea Reference Point', zorder=10, alpha=0.8)

        plt.title(f'Comparison of Pareto Fronts from Different Mutation Strategies [{DATASET}]')
        plt.xlabel('Total Cost ($)')
        plt.ylabel('Average Wait Time (hours)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_path = f'results/pareto_front_comparison_{DATASET}.png'
        plt.savefig(plot_path)
        print(f"\nSaved comparison plot to '{plot_path}'")

        # --- Metrics Table Calculation ---
        print("\n--- Quality Metrics Table ---\n")
        print(f"{'Variant':<20} | {'Hyperarea Ratio':<18} | {'Purity':<10} | {'Hausdorff Dist':<18} | {'Delta Spread':<15}")
        print("-" * 88)

        hv_ref = calculate_hyperarea(pref, ref_point)
        
        for data in all_run_data:
            front_np = np.array(data['pareto_front'])
            
            hv_approx = calculate_hyperarea(front_np, ref_point)
            hv_ratio = hv_approx / hv_ref if hv_ref > 0 else 0.0
            purity = calculate_purity(front_np, pref)
            hausdorff = calculate_averaged_hausdorff(front_np, pref)
            delta = calculate_delta_spread(front_np, pref)
                
            print(f"{data['variant_name']:<20} | {hv_ratio:<18.3f} | {purity:<10.3f} | {hausdorff:<18.2f} | {delta:<15.3f}")

    # =========================================================================
    # --- Analysis of the Most Balanced Solution's Improvement ---
    # =========================================================================
    print("\n\n--- Most Balanced Solution Improvement Analysis (vs Baselines) ---")
    print("Identifies the solution closest to the 'ideal' point (0,0) in normalized space for each front.\n")

    for data in all_run_data:
        front_np = np.array(data['pareto_front'])
        balanced_solution = find_most_balanced_solution(front_np)
        print(f"Variant: {data['variant_name']}")
        
        if balanced_solution is not None:
            cost, wait = balanced_solution
            if not data['baselines']:
                print("  (No baselines found)")
            else:
                for bname, bvals in data['baselines'].items():
                    cost_improv = (bvals['cost'] - cost) / bvals['cost'] * 100 if bvals.get('cost') else 0.0
                    wait_improv = (bvals['wait'] - wait) / bvals['wait'] * 100 if bvals.get('wait') else 0.0
                    print(f"  - Baseline '{bname}': Cost Impr: {cost_improv:.2f}% | Wait Impr: {wait_improv:.2f}%")
        else:
            print("  (No balanced solution found)")
        print("")

if __name__ == "__main__":
    main()