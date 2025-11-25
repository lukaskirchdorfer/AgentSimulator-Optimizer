import os
import math
import json
from datetime import datetime
import numpy as np
import scipy.stats as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def store_preprocessed_data(df_train, df_test, df_val, data_dir):
    print(data_dir)
    os.system(f"mkdir {data_dir}")
    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)

    path_to_train_file = os.path.join(data_dir,"train_preprocessed.csv")
    df_train_without_end_activity = df_train.copy()
    df_train_without_end_activity = df_train_without_end_activity[df_train_without_end_activity['activity_name'] != 'zzz_end']
    df_train_without_end_activity.to_csv(path_to_train_file, index=False)

    # save test data
    if df_test is not None:
        path_to_test_file = os.path.join(data_dir,"test_preprocessed.csv")
        df_test_without_end_activity = df_test.copy()
        df_test_without_end_activity = df_test_without_end_activity[df_test_without_end_activity['activity_name'] != 'zzz_end']
        df_test_without_end_activity.to_csv(path_to_test_file, index=False)

    return df_train_without_end_activity


def store_simulated_log(data_dir, simulated_log, index):
    path_to_file = os.path.join(data_dir,f"simulated_log_{index}.csv")
    simulated_log.to_csv(path_to_file, index=False)
    print(f"Simulated logs are stored in {path_to_file}")


def sample_from_distribution(distribution):
    if distribution.type.value == "expon":
        scale = distribution.mean - distribution.min
        if scale < 0.0:
            print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
            scale = distribution.mean
        sample = st.expon.rvs(loc=distribution.min, scale=scale, size=1)
    elif distribution.type.value == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(
            pow(distribution.mean, 2) / distribution.var,
            loc=0,
            scale=distribution.var / distribution.mean,
            size=1,
        )
    elif distribution.type.value == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=1)
    elif distribution.type.value == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=1)
    elif distribution.type.value == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
    elif distribution.type.value == "fix":
        sample = [distribution.mean] * 1

    return sample[0]

def generate_costs_file(df_train, baseline_parameters, dataset):
    # Build agent performance profiles
    activity_durations_dict = baseline_parameters.get('activity_durations_dict', {})
    agent_activity_mapping = baseline_parameters.get('agent_activity_mapping', {})

    # Prepare per-agent features: specialization (1/num_activities) and efficiency (1/mean_duration)
    agent_ids = sorted(list(activity_durations_dict.keys()))
    if not agent_ids:
        return {}

    features = []  # in original scale for post-processing
    perf_scores = []  # combined performance scores used for tier ranking

    epsilon = 1e-9
    for agent in agent_ids:
        activities = agent_activity_mapping.get(agent, list(activity_durations_dict.get(agent, {}).keys()))
        # Filter out artificial end activity if present
        activities = [a for a in activities if a != 'zzz_end']
        num_activities = max(1, len(activities))

        # Compute mean duration across activities for this agent using distribution means
        means = []
        for act, dist in activity_durations_dict.get(agent, {}).items():
            if act == 'zzz_end':
                continue
            if hasattr(dist, 'mean') and dist.mean is not None:
                means.append(float(dist.mean))
        mean_duration = float(np.mean(means)) if len(means) > 0 else 0.0

        # Specialization: higher when agent is more specialized (fewer activities)
        specialization = 1.0 / float(num_activities)
        # Efficiency: higher when mean duration is lower (avoid divide-by-zero)
        efficiency = 1.0 / float(mean_duration + epsilon)

        # Store raw feature vector
        features.append([specialization, efficiency])
        # Combined score for ranking clusters later
        perf_scores.append(0.5 * specialization + 0.5 * efficiency)

    features = np.array(features, dtype=float)
    perf_scores = np.array(perf_scores, dtype=float)

    # Scale features for clustering if possible
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # KMeans clustering into 5 tiers
    k = 5
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Rank clusters by average performance score (higher => higher tier and higher cost)
    cluster_to_mean_perf = {}
    for c in range(k):
        mask = labels == c
        cluster_to_mean_perf[c] = float(perf_scores[mask].mean()) if np.any(mask) else -np.inf
    ranked_clusters = sorted(cluster_to_mean_perf.items(), key=lambda kv: kv[1], reverse=True)
    cluster_rank_to_tier = {cluster: rank + 1 for rank, (cluster, _) in enumerate(ranked_clusters)}

    # Map tiers to hourly cost ranges
    tier_to_range = {
        1: (10, 25),   # lowest cost
        2: (26, 40),
        3: (41, 55),
        4: (56, 75),
        5: (76, 90),   # highest cost
    }

    # Sample cost per agent according to its tier
    rng = np.random.default_rng(42)
    resource_costs = {}
    for agent, label in zip(agent_ids, labels):
        tier = cluster_rank_to_tier[label]
        lo, hi = tier_to_range[tier]
        sampled_cost = int(rng.integers(lo, hi + 1))
        resource_costs[str(agent)] = sampled_cost

    # Ensure costs directory exists and store file
    costs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'costs')
    try:
        os.makedirs(costs_dir, exist_ok=True)
    except Exception:
        pass

    out_path = os.path.join(costs_dir, f"costs_{dataset}.json")
    try:
        with open(out_path, 'w') as f:
            json.dump(resource_costs, f, indent=4)
        print(f"Generated agent costs saved to {out_path}")
    except Exception as e:
        print(f"Warning: failed to write costs file to {out_path}: {e}")

    return resource_costs