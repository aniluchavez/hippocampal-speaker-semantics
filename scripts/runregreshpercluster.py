#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('hostnamectl')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script runregreshpercluster.ipynb')


# In[18]:


# === 1. Imports and Setup ===
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# === File paths ===
embedding_path = "/projects/bhayden/anilu/Language_docs/Regression/PTYEU_task147_words_english_only/PTYEU_task147_aligned_embeddings_withNP.csv"
duration_path = "/projects/bhayden/anilu/Language_docs/Regression/PTYEU_task147_words_english_only/PTYEU_task147_filtered_used_rows_withNP.xlsx"
cluster_path = "/projects/bhayden/anilu/Language_docs/Regression/ClusterLabels/PTYEU_task147_clusID_labels.xlsx"
spike_base_dir = "/projects/bhayden/anilu/Language_docs/Regression/output_PTYEU_task147_english_only"
speakers = [f"Speaker{i}" for i in range(1, 8)]
regions = ["ACC", "hippocampus"]

# === Load embedding metadata and durations ===
meta = pd.read_csv(embedding_path)
meta["Parsed_Embedding"] = meta["Embedding"].apply(eval)  # convert string to list
durations = pd.read_excel(duration_path)["Duration"]
meta["Duration"] = durations

# === Load cluster labels ===
clus_df = pd.read_excel(cluster_path)

# === Merge using row index for perfect alignment ===
clus_df["RowIndex"] = clus_df.index
meta["RowIndex"] = meta.index

# Only merge NewClusID back into meta using RowIndex
meta = meta.merge(clus_df[["RowIndex", "NewClusID"]], on="RowIndex", how="left")
meta = meta.dropna(subset=["NewClusID"]).reset_index(drop=True)
meta["NewClusID"] = meta["NewClusID"].astype(int)

# === Separate into self vs other ===
meta_self = meta[meta["Speaker"] == "SPK1"].copy().reset_index(drop=True)
meta_other = meta[meta["Speaker"] != "SPK1"].copy().reset_index(drop=True)

# === 2. Load Spike Data ===
def load_spike_data(base_dir, speakers, regions):
    spike_data = {}
    for speaker in speakers:
        spike_data[speaker] = {}
        for region in regions:
            path = f"{base_dir}/{speaker}/{region}_spike_sum.npy"
            try:
                spike_data[speaker][region] = np.load(path)
            except FileNotFoundError:
                print(f"[WARN] Missing: {path}")
    return spike_data

# === 3. Build X/Y by Cluster ===
def build_XY_clusters(meta_df, spike_data, region, n_components=10, n_clusters=15, min_trials=1):
    Y, X, cluster_ids = [], [], []
    speaker_counters = {spk: 0 for spk in spike_data}

    for _, row in meta_df.iterrows():
        spk = row["Speaker"]
        speaker_name = f"Speaker{spk.replace('SPK','')}"
        if speaker_name not in spike_data or region not in spike_data[speaker_name]:
            continue
        spikes = spike_data[speaker_name][region]
        idx = speaker_counters[speaker_name]
        if idx >= spikes.shape[0]:
            continue

        Y.append(spikes[idx])
        emb = np.array(row["Parsed_Embedding"])
        dur = row["Duration"]
        X.append((emb, dur))
        cluster_ids.append(row["NewClusID"])
        speaker_counters[speaker_name] += 1

    if not Y:
        print(f"[WARN] No valid trials for region={region}")
        dummy_X = np.empty((0, n_components * 2 + 1))
        dummy_Y = np.empty((0, 1))
        return {i: dummy_X for i in range(n_clusters)}, {i: dummy_Y for i in range(n_clusters)}

    # Build X matrix
    embeddings = np.vstack([e for e, d in X])
    durations = np.array([d for e, d in X]).reshape(-1, 1)
    pcs = PCA(n_components=n_components).fit_transform(embeddings)
    interactions = pcs * durations
    X_all = StandardScaler().fit_transform(np.hstack([pcs, durations, interactions]))
    Y_all = np.vstack(Y)
    cluster_ids = np.array(cluster_ids)

    # Split into clusters
    X_clusters, Y_clusters = {}, {}
    for clus_id in range(n_clusters):
        mask = cluster_ids == clus_id
        if mask.sum() >= min_trials:
            X_clusters[clus_id] = X_all[mask]
            Y_clusters[clus_id] = Y_all[mask]
        else:
            X_clusters[clus_id] = np.empty((0, X_all.shape[1]))
            Y_clusters[clus_id] = np.empty((0, Y_all.shape[1]))
    return X_clusters, Y_clusters


# In[20]:


# === 4. Load spikes and run for ACC ===
spike_data = load_spike_data(spike_base_dir, speakers, regions)

# === 5. Run design matrix builder per condition and region ===
X_clusters_self_ACC, Y_clusters_self_ACC = build_XY_clusters(meta_self, spike_data, n_components=20,region="ACC")
X_clusters_other_ACC, Y_clusters_other_ACC = build_XY_clusters(meta_other, spike_data, n_components=20,region="ACC")

# === 6. Summary of row counts ===
total_X_self = sum(x.shape[0] for x in X_clusters_self_ACC.values())
total_Y_self = sum(y.shape[0] for y in Y_clusters_self_ACC.values())
total_X_other = sum(x.shape[0] for x in X_clusters_other_ACC.values())
total_Y_other = sum(y.shape[0] for y in Y_clusters_other_ACC.values())

print(f"✅ SPK1 trials in meta_self:  {len(meta_self)}")
print(f"✅ Other trials in meta_other: {len(meta_other)}\n")

print(f"📊 ACC region:")
print(f"   Total SPK1 X rows (clustered):  {total_X_self}")
print(f"   Total SPK1 Y rows (clustered):  {total_Y_self}")
print(f"   Total OTHER X rows (clustered): {total_X_other}")
print(f"   Total OTHER Y rows (clustered): {total_Y_other}")

print("📦 Cluster size summary (ACC region):")
for clus_id in range(15):  # first 5 clusters
    x_s = X_clusters_self_ACC.get(clus_id, np.empty((0,)))
    y_s = Y_clusters_self_ACC.get(clus_id, np.empty((0,)))
    x_o = X_clusters_other_ACC.get(clus_id, np.empty((0,)))
    y_o = Y_clusters_other_ACC.get(clus_id, np.empty((0,)))
    
    print(f"Cluster {clus_id:2d} | SPK1: X={x_s.shape}, Y={y_s.shape} | Other: X={x_o.shape}, Y={y_o.shape}")


# In[21]:


# === Build X_dict and Y_dict across all regions/conditions ===
X_dict = {}
Y_dict = {}

for region in ["ACC", "hippocampus"]:
    print(f"📦 Processing region: {region}")
    
    X_self, Y_self = build_XY_clusters(meta_self, spike_data, n_components=100,region=region, n_clusters=15)
    X_other, Y_other = build_XY_clusters(meta_other, spike_data, n_components=100,region=region, n_clusters=15)
    
    X_dict[f"{region}_self"] = X_self
    Y_dict[f"{region}_self"] = Y_self
    X_dict[f"{region}_other"] = X_other
    Y_dict[f"{region}_other"] = Y_other

print("\n✅ Finished building X_dict and Y_dict.")

# Print shapes for the first 5 clusters of ACC_self
print("\n📊 ACC_self sample cluster sizes:")
for i in range(15):
    X = X_dict["ACC_other"].get(i, np.empty((0,)))
    Y = Y_dict["ACC_other"].get(i, np.empty((0,)))
    print(f"Cluster {i:2d} → X: {X.shape}, Y: {Y.shape}")



# 

# In[22]:


# === Regression: Poisson Ridge with Clustering Support ===
# === Regression: Poisson Ridge with Clustering Support ===
from joblib import Parallel, delayed
from tqdm import tqdm
def run_poisson_ridge(X, Y, patient_id, neuron_idx, region_name,
                      results_dir="./results",
                      n_iterations=10, test_size=0.3, use_clusters=False):
    
    import os
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    from scipy.special import gammaln
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore")


    if use_clusters:
        results_dir = os.path.join(results_dir, "clusterwise")
    else:
        results_dir = os.path.join(results_dir, "allwords")

    os.makedirs(results_dir, exist_ok=True)

    n_words, n_neurons = Y.shape
    y = Y[:, neuron_idx]

    if len(y) < 5:
        print(f"[SKIP] Not enough samples to fit CV for neuron {neuron_idx} in {region_name}")
        return pd.DataFrame(), pd.DataFrame()

    model = PoissonRegressor(max_iter=1000)
    param_grid = {'alpha': np.logspace(-3, 3, 30)}
    grid = GridSearchCV(model, param_grid, cv=min(5, len(y)), scoring='neg_mean_poisson_deviance', n_jobs=1)
    grid.fit(X, y)
    best_alpha = grid.best_params_['alpha']

    all_results = []
    ridge_coeff_accumulator = []
    ridge_support_flags = set()
    neuron_predictor_records = []

    y_true_all, y_pred_all = [], []
    train_lls, test_lls = [], []
    train_corrs, test_corrs = [] ,[]

    for iter_idx in range(n_iterations):
        split_point = int((1 - test_size) * n_words)
        offset = (iter_idx * int(n_words * test_size)) % n_words
        idx_range = np.roll(np.arange(n_words), -offset)
        train_idx = idx_range[:split_point]
        test_idx = idx_range[split_point:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        ridge_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        ridge_model.fit(X_train, y_train)
        pred_train = np.clip(ridge_model.predict(X_train), 1e-10, None)
        pred_test = np.clip(ridge_model.predict(X_test), 1e-10, None)

        ll_train = np.sum(y_train * np.log(pred_train) - pred_train - gammaln(y_train + 1))
        ll_real = np.sum(y_test * np.log(pred_test) - pred_test - gammaln(y_test + 1))

        if len(y_train) >= 2 and len(pred_train) >= 2:
            r_train, _ = pearsonr(y_train, pred_train)
        else:
            r_train = np.nan

        if len(y_test) >= 2 and len(pred_test) >= 2:
            r_test, corr_p = pearsonr(y_test, pred_test)
        else:
            r_test = np.nan
            corr_p = np.nan

        r2 = r2_score(y_test, pred_test) if len(y_test) >= 2 else np.nan
        n, k = len(y_test), X_test.shape[1]
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1) if n > k + 1 else r2

        y_true_all.extend(y_test)
        y_pred_all.extend(pred_test)
        train_lls.append(ll_train)
        test_lls.append(ll_real)
        train_corrs.append(r_train)
        test_corrs.append(r_test)
        ridge_coeff_accumulator.append(ridge_model.coef_)

        y_train_shuf = np.random.permutation(y_train)
        null_model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
        null_model.fit(X_train, y_train_shuf)
        null_pred = np.clip(null_model.predict(X_test), 1e-10, None)
        ll_shuf = np.sum(y_test * np.log(null_pred) - null_pred - gammaln(y_test + 1))

        if len(y_test) >= 2 and len(null_pred) >= 2:
            corr_shuf, corr_p_shuf = pearsonr(y_test, null_pred)
            r2_shuf = r2_score(y_test, null_pred)
        else:
            corr_shuf, corr_p_shuf, r2_shuf = np.nan, np.nan, np.nan

        adj_r2_shuf = 1 - ((1 - r2_shuf) * (n - 1)) / (n - k - 1) if n > k + 1 and r2_shuf is not np.nan else r2_shuf
        ll_diff = ll_real - ll_shuf

        all_results.append({
            'neuron': neuron_idx, 'iteration': iter_idx, 'best_alpha': best_alpha,
            'll_real': ll_real, 'll_shuf': ll_shuf, 'll_diff': ll_diff,
            'adj_r2': adj_r2, 'adj_r2_shuf': adj_r2_shuf,
            'corr': r_test, 'corr_p': corr_p, 'corr_shuf': corr_shuf, 'corr_p_shuf': corr_p_shuf
        })

        try:
            glm = sm.GLM(y_train, X_train, family=sm.families.Poisson())
            result = glm.fit()
            for idx, (coef, pval) in enumerate(zip(result.params, result.pvalues)):
                if pval < 0.05:
                    ridge_support_flags.add(idx)
                    neuron_predictor_records.append((idx, coef, pval))
        except:
            continue

    # === Save Outputs ===
    df_results = pd.DataFrame(all_results)
    coef_mean = np.mean(np.vstack(ridge_coeff_accumulator), axis=0)

    ridge_df = pd.DataFrame({
        'neuron': neuron_idx,
        'predictor_index': np.arange(len(coef_mean)),
        'mean_ridge_coef': coef_mean,
        'ridge_support': [1 if i in ridge_support_flags else 0 for i in range(len(coef_mean))]
    })

    return df_results, ridge_df


def run_cluster_parallel(X, Y, region, cluster_id, condition, patient_id, results_dir, n_iterations=10, use_parallel=False):
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print(f"[SKIP] Cluster {cluster_id} - {condition} - {region}: Empty matrix")
        return pd.DataFrame()

    n_neurons = Y.shape[1]

    if use_parallel:
        outputs = Parallel(n_jobs=-1)(
            delayed(run_poisson_ridge)(
                X, Y,
                patient_id=patient_id,
                neuron_idx=neuron_idx,
                region_name=f"{region}_{condition}_clus{cluster_id}",
                results_dir=results_dir,
                n_iterations=n_iterations,
                use_clusters=True
            ) for neuron_idx in tqdm(range(n_neurons), desc=f"{region}-{condition}-cluster{cluster_id}")
        )
    else:
        outputs = []
        for neuron_idx in tqdm(range(n_neurons), desc=f"{region}-{condition}-cluster{cluster_id} [serial]"):
            out = run_poisson_ridge(
                X, Y,
                patient_id=patient_id,
                neuron_idx=neuron_idx,
                region_name=f"{region}_{condition}_clus{cluster_id}",
                results_dir=results_dir,
                n_iterations=n_iterations,
                use_clusters=True
            )
            outputs.append(out)

    all_coefs = [out[1].assign(neuron=n) for n, out in enumerate(outputs)]
    return pd.concat(all_coefs, ignore_index=True)

def run_all_conditions(X_dict, Y_dict, patient_id, regions, conditions=["self", "other"], n_clusters=15, use_parallel=True, results_dir="./results"):
    all_outputs = []

    for region in regions:
        for condition in conditions:
            key = f"{region}_{condition}"
            if key not in X_dict or key not in Y_dict:
                continue

            X_clusters = X_dict[key]
            Y_clusters = Y_dict[key]

            for clus_id in range(n_clusters):
                if clus_id not in X_clusters or clus_id not in Y_clusters:
                    continue
                if X_clusters[clus_id].shape[0] == 0 or Y_clusters[clus_id].shape[0] == 0:
                    print(f"[SKIP] {region} - {condition} - cluster {clus_id}: no data")
                    continue

                X = X_clusters[clus_id]
                Y = Y_clusters[clus_id]

                result_df = run_cluster_parallel(
                    X, Y,
                    region=region,
                    cluster_id=clus_id,
                    condition=condition,
                    patient_id=patient_id,
                    results_dir=results_dir,
                    use_parallel=use_parallel
                )
                if not result_df.empty:
                    result_df["region"] = region
                    result_df["condition"] = condition
                    result_df["cluster_id"] = clus_id
                    result_df["patient"] = patient_id
                    all_outputs.append(result_df)

    return pd.concat(all_outputs, ignore_index=True) if all_outputs else pd.DataFrame()


# In[23]:


import os

# === 1. Set Patient Info and Paths ===
patient_id = "PTYEU147"
results_dir = f"/projects/bhayden/anilu/Language_docs/Regression/RegressionRESULTS/{patient_id}"  # <-- organize per patient

# === 2. Define regions and settings ===
regions = ["ACC", "hippocampus"]
use_parallel = True
n_iterations = 10

# === 3. Run regression model across all clusters/conditions/regions ===
regression_df = run_all_conditions(
    X_dict=X_dict,
    Y_dict=Y_dict,
    patient_id=patient_id,
    regions=regions,
    conditions=["self", "other"],
    n_clusters=15,
    use_parallel=use_parallel,
    results_dir=results_dir
)

# === 4. Save neuron × predictor matrix ===
summary_path = os.path.join(results_dir, f"summary_all_neuron_predictors_{patient_id}.csv")
regression_df.to_csv(summary_path, index=False)

# === 5. Show storage locations ===
print("✅ Done! Regression results saved.")
print(f"📁 Summary CSV: {summary_path}")
print(f"📁 Summary CSV: {summary_path}")

# Optional: show number of rows/neurons
print(f"🧠 Total results: {len(regression_df)} rows (across neurons/clusters/regions)")



# In[24]:


from scipy.spatial.distance import cosine

def compute_cosine_distances_by_cluster(all_results_df):
    """
    Compute cosine distances between self and other neurons for each cluster+region.

    Returns a summary DataFrame.
    """
    all_summary = []

    grouped = all_results_df.groupby(["region", "cluster_id"])

    for (region, cluster_id), df_group in grouped:
        df_self = df_group[df_group["condition"] == "self"].sort_values("neuron")
        df_other = df_group[df_group["condition"] == "other"].sort_values("neuron")

        # Skip if any condition missing or empty
        if df_self.empty or df_other.empty:
            continue

        # Pivot to wide format (neuron x predictors)
        pivot_self = df_self.pivot(index="neuron", columns="predictor_index", values="mean_ridge_coef")
        pivot_other = df_other.pivot(index="neuron", columns="predictor_index", values="mean_ridge_coef")

        # Intersect neurons to align
        shared_neurons = pivot_self.index.intersection(pivot_other.index)
        if len(shared_neurons) < 2:
            continue  # skip underpowered cluster

        mat_self = pivot_self.loc[shared_neurons].values
        mat_other = pivot_other.loc[shared_neurons].values

        # Compute cosine distance per neuron
        cos_dists = [cosine(vec_self, vec_other)
                     for vec_self, vec_other in zip(mat_self, mat_other)]

        all_summary.append({
            "region": region,
            "cluster_id": cluster_id,
            "n_neurons": len(shared_neurons),
            "mean_cosine_distance": np.mean(cos_dists),
            "std_cosine_distance": np.std(cos_dists),
        })

    return pd.DataFrame(all_summary)


# In[25]:


from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

def compute_cosine_distances_by_cluster(df):
    summary = []

    grouped = df.groupby(["region", "cluster_id", "condition", "neuron"])
    
    # Get beta vectors: [region][cluster][condition][neuron] → vector
    nested_dict = {}
    for (region, cluster_id, condition, neuron), subdf in grouped:
        vec = subdf.sort_values("predictor_index")["mean_ridge_coef"].values
        key = (region, cluster_id, neuron)
        if key not in nested_dict:
            nested_dict[key] = {}
        nested_dict[key][condition] = vec

    # Compute cosine distance for each neuron between self and other
    for (region, cluster_id, neuron), cond_dict in nested_dict.items():
        if "self" in cond_dict and "other" in cond_dict:
            v1, v2 = cond_dict["self"], cond_dict["other"]
            if len(v1) == len(v2):
                dist = cosine_distances([v1], [v2])[0][0]
                summary.append({
                    "region": region,
                    "cluster_id": cluster_id,
                    "neuron": neuron,
                    "cosine_distance": dist
                })

    return pd.DataFrame(summary)


# In[26]:


cosine_summary = compute_cosine_distances_by_cluster(regression_df)

# Average cosine distances per cluster
avg_distances = (
    cosine_summary
    .groupby(["region", "cluster_id"])["cosine_distance"]
    .mean()
    .reset_index()
)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_distances, x="cluster_id", y="cosine_distance", hue="region")
plt.title("Average Cosine Distance Between Self and Other Betas per Cluster")
plt.xlabel("Semantic Cluster ID")
plt.ylabel("Cosine Distance (Self vs Other)")
plt.legend(title="Region")
plt.tight_layout()
plt.show()


# 

# In[32]:


# === Batch Run: Regression Across Patients ===
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# === CONFIG ===
patient_ids = [
    # "PTYFC_task28",
    # "PTYEU_task147",
    # "PTYFF_task17",
    # "PTYFI_task81",
    # "PTYFG_task18",
    "PTYEZ_task60",
    "PTYEV_task37"
]
regions = ["hippocampus", "ACC"]
speakers = [f"Speaker{i}" for i in range(1, 8)]
target_speaker = "SPK1"
n_iterations = 10
use_parallel = True

# === IMPORT SHARED FUNCTIONS ===
# Assume you have imported:
# - get_self_and_other_features
# - load_spike_data
# - build_Y_matrices
# - run_all_conditions

all_regression_results = []

for patient_id in patient_ids:
    print(f"\n==============================")
    print(f"🚀 Running patient: {patient_id}")
    print(f"==============================")

    try:
        # --- Set file paths ---
        embedding_file = f"/projects/bhayden/anilu/Language_docs/Regression/{patient_id}_words_english_only/{patient_id}_aligned_embeddings_withNP.csv"
        duration_file  = f"/projects/bhayden/anilu/Language_docs/Regression/{patient_id}_words_english_only/{patient_id}_filtered_used_rows_withNP.xlsx"
        cluster_path   = f"/projects/bhayden/anilu/Language_docs/Regression/ClusterLabels/{patient_id}_clusID_labels.xlsx"
        spike_base_dir = f"/projects/bhayden/anilu/Language_docs/Regression/output_{patient_id}_english_only"
        results_dir    = f"/projects/bhayden/anilu/Language_docs/Regression/RegressionRESULTS/{patient_id}"

        # --- Load Data ---
        meta = pd.read_csv(embedding_file)
        meta["Parsed_Embedding"] = meta["Embedding"].apply(eval)
        durations = pd.read_excel(duration_file)["Duration"]
        meta["Duration"] = durations

        clus_df = pd.read_excel(cluster_path)
        clus_df["RowIndex"] = clus_df.index
        meta["RowIndex"] = meta.index
        meta = meta.merge(clus_df[["RowIndex", "NewClusID"]], on="RowIndex", how="left")
        meta = meta.dropna(subset=["NewClusID"]).reset_index(drop=True)
        meta["NewClusID"] = meta["NewClusID"].astype(int)

        meta_self = meta[meta["Speaker"] == target_speaker].copy().reset_index(drop=True)
        meta_other = meta[meta["Speaker"] != target_speaker].copy().reset_index(drop=True)

        spike_data = load_spike_data(spike_base_dir, speakers, regions)

        # --- Build design matrices ---
        X_dict, Y_dict = {}, {}
        for region in regions:
            X_self, Y_self = build_XY_clusters(meta_self, spike_data, region=region, n_components=100, n_clusters=15)
            X_other, Y_other = build_XY_clusters(meta_other, spike_data, region=region, n_components=100, n_clusters=15)

            X_dict[f"{region}_self"] = X_self
            Y_dict[f"{region}_self"] = Y_self
            X_dict[f"{region}_other"] = X_other
            Y_dict[f"{region}_other"] = Y_other

        # --- Run Regression ---
        regression_df = run_all_conditions(
            X_dict=X_dict,
            Y_dict=Y_dict,
            patient_id=patient_id,
            regions=regions,
            conditions=["self", "other"],
            n_clusters=15,
            use_parallel=use_parallel,
            results_dir=results_dir
        )

        # --- Save individual patient result ---
        summary_path = os.path.join(results_dir, f"summary_all_neuron_predictors_{patient_id}.csv")
        regression_df.to_csv(summary_path, index=False)
        all_regression_results.append(regression_df)

        print(f"✅ Saved: {summary_path}")

    except Exception as e:
        print(f"❌ Error processing {patient_id}: {e}")
        continue

# === Optionally Save Combined Data ===
combined_df = pd.concat(all_regression_results, ignore_index=True)
combined_df.to_csv("/projects/bhayden/anilu/Language_docs/Regression/RegressionRESULTS/all_patients_combined_regression.csv", index=False)
print("\n✅✅✅ BATCH RUN COMPLETE ✅✅✅")


# In[48]:


import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG: List of patients ===
patient_ids = [
    "PTYFC_task28", "PTYEU_task147", "PTYFF_task17",
    "PTYFI_task81", "PTYFG_task18", "PTYEZ_task60", "PTYEV_task37"
]
results_base = "/projects/bhayden/anilu/Language_docs/Regression/RegressionRESULTS"

# === Load and concatenate regression summaries ===
all_dfs = []
for pid in patient_ids:
    path = os.path.join(results_base, pid, f"summary_all_neuron_predictors_{pid}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["patient"] = pid
        all_dfs.append(df)
    else:
        print(f"⚠️ Missing: {path}")
if not all_dfs:
    raise ValueError("No valid regression outputs found.")
df_all = pd.concat(all_dfs, ignore_index=True)

# === Compute average beta vectors and cosine distances (deduplicated) ===
grouped = df_all.groupby(["region", "cluster_id", "condition", "predictor_index"])
beta_vectors = {}
for (region, cluster_id, condition, pred_idx), subdf in grouped:
    key = (region, cluster_id, condition)
    beta_vectors.setdefault(key, []).append(subdf["mean_ridge_coef"].mean())

cosine_results = []
for (region, cluster_id, condition), vec in beta_vectors.items():
    if condition != "self":
        continue  # compute distance only once per cluster

    counterpart = (region, cluster_id, "other")
    if counterpart in beta_vectors:
        v1 = np.array(vec)
        v2 = np.array(beta_vectors[counterpart])
        if len(v1) == len(v2):
            dist = cosine_distances([v1], [v2])[0][0]
            cosine_results.append({
                "region": region,
                "cluster_id": cluster_id,
                "cosine_distance": dist
            })

cosine_df = pd.DataFrame(cosine_results)

# === Plot: Sorted rainbow barplot per region ===
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cosine_per_region_single_bars(cosine_df):
    sns.set(style="white", font_scale=1.3)

    # === Attach human-readable semantic categories ===
    categories = [
        "Physical Objects / Food", "Body Parts", "Descriptive Traits / Adjectives", "Identity",
        "Mental States / Abstract Concepts", "Emotions / Feelings", "Medical / Health",
        "Geography / Places / Environments", "Social Relationships / Family", "Education / Work / Institutions",
        "Actions / Movements / Verbs", "Temporal / Quantities / Time", "Media / Tech / Communication",
        "Cultural / Names / Nationalities", "Function Words / Slang / Interjections"
    ]
    category_mapping = dict(zip(range(15), categories))
    cosine_df["category"] = cosine_df["cluster_id"].map(category_mapping)

    for region in cosine_df["region"].unique():
        region_data = (
            cosine_df[cosine_df["region"] == region]
            .groupby("category", as_index=False)["cosine_distance"]
            .mean()
            .sort_values("cosine_distance", ascending=True)
        )

        # Create pastel rainbow palette by blending hsv with white
        base_palette = sns.color_palette("hsv", len(region_data))
                # Create pastel rainbow palette by blending hsv with white
        base_palette = sns.color_palette("hsv", len(region_data))
        pastel_palette = [((r + 1) / 2, (g + 1) / 2, (b + 1) / 2) for r, g, b in base_palette]


        plt.figure(figsize=(16, 7))  # Wider and taller
        ax = sns.barplot(data=region_data, x="category", y="cosine_distance", palette=pastel_palette)

        ax.set_title(f"Region: {region} — Cosine Distance by Semantic Category")
        ax.set_xlabel("Semantic Category")
        ax.set_ylabel("Cosine Distance (Self vs Other)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        plt.tight_layout()
        plt.show()


plot_cosine_per_region_single_bars(cosine_df)

