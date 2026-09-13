import pandas as pd
import numpy as np
import torch
import faiss
import os
import re
import time
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss, roc_auc_score
from dataset import CallDataset
from embeddings import Embeddings
from resnet_ablation import ResNetAblation

def read_train_time(txt_path, model_name, split_prefix):
    try:
        with open(txt_path, "r", encoding = "utf-8") as f:
            for line in f:
                if model_name in line and f"Segment: {split_prefix}" in line:
                    match = re.search(r'\(([\d\.]+)\s*s\)', line)
                    if match:
                        return float(match.group(1))
    except FileNotFoundError:
        pass
    return 0.0

def calculate_ece(y_true, y_prob, n_bins = 10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids[binids == n_bins] = n_bins - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_diff = np.abs(np.mean(y_prob[mask]) - np.mean(y_true[mask]))
            ece += (np.sum(mask) / len(y_prob)) * prob_diff
    return ece

def log_metrics(results, model_name, segment, y_true, y_prob, time_seconds):
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    results.append({
        "Model": model_name, 
        "Segment": segment, 
        "Brier": brier, 
        "AUC": auc, 
        "ECE": ece, 
        "Time(s)": time_seconds
    })
    return results

def evaluate_lens(model, df_train, df_test, k_neighbors, temp, batch_size, device):
    dataloader_train = DataLoader(df_train, batch_size = batch_size, shuffle = False)
    dataloader_test = DataLoader(df_test, batch_size = batch_size, shuffle = False)
    
    model.eval()
    
    embs_train, labels_train = [], []
    with torch.no_grad():
        for x, y in dataloader_train:
            embs_train.append(model(x.to(device)).cpu().numpy())
            labels_train.append(y.numpy())
    embs_train = np.vstack(embs_train).astype(np.float32)
    labels_train = np.concatenate(labels_train)

    index = faiss.IndexFlatIP(embs_train.shape[1])
    index.add(embs_train)

    embs_test, labels_test = [], []
    with torch.no_grad():
        for x, y in dataloader_test:
            embs_test.append(model(x.to(device)).cpu().numpy())
            labels_test.append(y.numpy())
    embs_test = np.vstack(embs_test).astype(np.float32)
    labels_test = np.concatenate(labels_test)

    # Use the neighbors' outcomes to estimate the probability empirically.
    distances, indices = index.search(embs_test, k_neighbors)
    weights = torch.nn.functional.softmax(torch.tensor(distances) / temp, dim = 1).numpy()
    outcomes = labels_train[indices]
    probabilities = np.sum(weights * outcomes, axis = 1)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    
    return probabilities, labels_test

def evaluate_ablation(model, df_test, batch_size, device):
    dataloader_test = DataLoader(df_test, batch_size = batch_size, shuffle = False)
    model.eval()
    probabilities_list = []
    
    with torch.no_grad():
        for x, _ in dataloader_test:
            logits = model(x.to(device)).view(-1)
            probs = torch.sigmoid(logits)
            probabilities_list.extend(probs.cpu().numpy())
            
    return np.array(probabilities_list)

def evaluate_xgboost(df_train, df_test, iterations, cat_cols_base, seed = 42):
    X_train = df_train.drop(columns = ["y"]).copy()
    y_train = df_train["y"].values
    X_test = df_test.drop(columns = ["y"]).copy()
    
    for col in cat_cols_base:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    # XGBoost trained on the same splits serves as a robust tabular baseline to compare against LENS[cite: 1].
    model = xgb.XGBClassifier(
        n_estimators = iterations, 
        subsample = 0.8, 
        colsample_bytree = 0.8,
        learning_rate = 0.05, 
        max_depth = 3,
        min_child_weight = 5,
        tree_method = "hist",
        enable_categorical = True, 
        objective = "binary:logistic", 
        eval_metric = "rmse",
        random_state = seed,
        n_jobs = -1
    )
    model.fit(X_train, y_train, verbose = False)
    probs = model.predict_proba(X_test)[:, 1]
    
    return probs

def generate_charts(df_results):
    os.makedirs("results", exist_ok = True)
    
    segments = ["Global", "VIP (Contact)", "Masivo (No Contact)"]
    metrics = ["Brier", "ECE"]
    
    models_plot = [
        {"csv_name": "Standard ResNet", "label": "Semantic ResNet"},
        {"csv_name": "ResNet + LENS", "label": "Semantic ResNet + LENS"}
    ]
    
    color_map = {
        "Semantic ResNet": "lightblue",
        "Semantic ResNet + LENS": "royalblue"
    }

    for metric in metrics:
        save_path = os.path.join("results", metric)
        os.makedirs(save_path, exist_ok = True)

        for segment in segments:
            # Extract XGBoost baseline dynamically from the results DataFrame
            xgb_row = df_results[(df_results["Model"] == "XGBoost") & (df_results["Segment"] == segment)]
            xgb_val = xgb_row[metric].values[0] if not xgb_row.empty else 0.0

            data = []
            for mod in models_plot:
                row = df_results[(df_results["Model"] == mod["csv_name"]) & (df_results["Segment"] == segment)]
                if not row.empty:
                    val = row[metric].values[0]
                    time_val = row["Time(s)"].values[0]
                    label = mod["label"]
                    color = color_map[label]
                    data.append((val, time_val, label, color))
            
            data.sort(key = lambda x: x[0], reverse = True)
            
            if not data:
                continue

            max_y = max([d[0] for d in data] + [xgb_val])

            fig, ax = plt.subplots(figsize = (8, 6))
            fig.suptitle(f"{metric} Comparison - Segment: {segment}", fontsize = 14, fontweight = "bold")

            values = [d[0] for d in data]
            times = [d[1] for d in data]
            labels = [d[2] for d in data]
            colors = [d[3] for d in data]

            x = np.arange(len(labels))
            bars = ax.bar(x, values, color = colors, width = 0.6, edgecolor = "black")
            
            # The red dashed line represents the XGBoost baseline dynamically.
            if xgb_val > 0.0:
                ax.axhline(y = xgb_val, color = "red", linestyle = "--", linewidth = 2, label = "XGBoost Baseline")

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + (max_y * 0.01), 
                        f"{height:.4f}\n({times[i]:.0f}s)", 
                        ha = "center", va = "bottom", fontweight = "bold", color = "black")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation = 15, ha = "right")
            ax.legend(loc = "upper right")
            ax.set_ylim(0, max_y * 1.20)
            ax.set_ylabel(metric, fontsize = 12)

            clean_seg_name = segment.replace(" ", "_").replace("(", "").replace(")", "")
            file_name = os.path.join(save_path, f"chart_{metric}_{clean_seg_name}.png")
            
            plt.tight_layout()
            plt.savefig(file_name, format = "png", bbox_inches = "tight")
            plt.close()

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    configs = [
        {
            "segment": "VIP (Contact)",
            "train": "data/bank_contact_train.parquet",
            "val": "data/bank_contact_val.parquet",
            "test": "data/bank_contact_test.parquet",
            "prefix": "contact", "entity_dim": 32, "num_blocks": 2,
            "latent_dim": 512, "k": 63, "temp": 0.06, "batch_size": 128,
            "hidden_dim": 512, "xgb_iters": 40
        },
        {
            "segment": "Masivo (No Contact)",
            "train": "data/bank_nocontact_train.parquet",
            "val": "data/bank_nocontact_val.parquet",
            "test": "data/bank_nocontact_test.parquet",
            "prefix": "nocontact", "entity_dim": 16, "num_blocks": 3,
            "latent_dim": 1024, "k": 63, "temp": 0.07, "batch_size": 2048,
            "hidden_dim": 1024, "xgb_iters": 250
        }
    ]

    global_probs = {
        "XGBoost": [],
        "Standard ResNet": [],
        "ResNet + LENS": [], 
        "Labels": []
    }
    global_times = {
        "XGBoost": 0.0,
        "Standard ResNet": 0.0,
        "ResNet + LENS": 0.0
    }

    for cfg in configs:
        print(f"\n--- Segment {cfg['segment']} ---")
        
        # Load datasets for neural networks
        df_train = CallDataset(cfg["train"])
        df_test = CallDataset(cfg["test"])
        
        max_indices = torch.tensor(df_train.feature_dims, dtype = torch.long) - 1
        df_test.features = torch.clamp(df_test.features, min = 0)
        df_test.features = torch.minimum(df_test.features, max_indices)
        
        # Load raw pandas DataFrames for XGBoost
        raw_train_df = pd.read_parquet(cfg["train"])
        raw_test_df = pd.read_parquet(cfg["test"])
        
        labels = raw_test_df["y"].values
        global_probs["Labels"].extend(labels)

        # 1. XGBoost On-The-Fly Evaluation
        print("Evaluating XGBoost...")
        start_xgb = time.time()
        # Define cat_cols_base according to the dataset structure. Assuming all non-target columns are categorical.
        cat_cols_base = [col for col in raw_train_df.columns if col != "y"]
        probs_xgb = evaluate_xgboost(raw_train_df, raw_test_df, cfg["xgb_iters"], cat_cols_base)
        t_xgb = time.time() - start_xgb
        
        global_times["XGBoost"] += t_xgb
        global_probs["XGBoost"].extend(probs_xgb)
        results = log_metrics(results, "XGBoost", cfg["segment"], labels, probs_xgb, t_xgb)

        # 2. Standard ResNet Evaluation
        print("Evaluating Standard ResNet...")
        txt_path = "Models/training_times.txt"
        t_std = read_train_time(txt_path, "Standard ResNet", cfg["prefix"])
        std_path = f"Models/ResNet_Standard_weights_{cfg['prefix']}.pth"
        
        model_std = ResNetAblation(
            feature_dims = df_train.feature_dims, latent_dim = cfg["latent_dim"], 
            entity_dim = cfg["entity_dim"], dropout = 0.2, feature_dropout = 0.15, hidden_dim = cfg["hidden_dim"],
            num_blocks = cfg["num_blocks"]
        ).to(device)
        model_std.load_state_dict(torch.load(std_path, map_location = device))
        
        probs_std = evaluate_ablation(model_std, df_test, cfg["batch_size"], device)
        
        global_times["Standard ResNet"] += t_std
        global_probs["Standard ResNet"].extend(probs_std)
        results = log_metrics(results, "Standard ResNet", cfg["segment"], labels, probs_std, t_std)

        # 3. LENS Evaluation
        print("Evaluating ResNet + LENS...")
        t_lens = read_train_time(txt_path, "ResNet + LENS", cfg["prefix"])
        lens_path = f"Models/ResNet_LENS_weights_{cfg['prefix']}.pth"
        
        model_lens = Embeddings(
            feature_dims = df_train.feature_dims, latent_dim = cfg["latent_dim"], 
            entity_dim = cfg["entity_dim"], dropout = 0.2, feature_dropout = 0.15, hidden_dim = cfg["hidden_dim"],
            num_blocks = cfg["num_blocks"]
        ).to(device)
        model_lens.load_state_dict(torch.load(lens_path, map_location = device))
        
        probs_lens, _ = evaluate_lens(model_lens, df_train, df_test, cfg["k"], cfg["temp"], cfg["batch_size"], device)
        
        global_times["ResNet + LENS"] += t_lens
        global_probs["ResNet + LENS"].extend(probs_lens)
        results = log_metrics(results, "ResNet + LENS", cfg["segment"], labels, probs_lens, t_lens)

    lbls_arr = np.array(global_probs["Labels"])
    for model_name in ["XGBoost", "Standard ResNet", "ResNet + LENS"]:
        results = log_metrics(
            results, model_name, "Global", lbls_arr, np.array(global_probs[model_name]), global_times[model_name]
        )

    df_results = pd.DataFrame(results)
    order_dict = {"Global": 0, "VIP (Contact)": 1, "Masivo (No Contact)": 2}
    df_results['sort_val'] = df_results['Segment'].map(order_dict)
    df_results = df_results.sort_values(by = ["sort_val", "Model"]).drop(columns = ['sort_val'])
    
    df_results[["Brier", "AUC", "ECE", "Time(s)"]] = df_results[["Brier", "AUC", "ECE", "Time(s)"]].round(5)
    
    os.makedirs("results", exist_ok = True)
    df_results.to_csv("results/benchmark.csv", index = False)
    print("\n" + df_results.to_string(index = False))
    
    print("\nGenerating Brier and ECE charts...")
    generate_charts(df_results)
    print("Process finished!")

if __name__ == "__main__":
    run_experiment()