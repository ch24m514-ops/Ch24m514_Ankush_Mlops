import json, argparse, numpy as np, pandas as pd, os

def psi(expected, actual, buckets=10, eps=1e-6):
    # Population Stability Index using quantile bins of expected
    edges = np.quantile(expected, np.linspace(0,1,buckets+1))
    edges[0] = -np.inf; edges[-1] = np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = e_counts / (e_counts.sum()+eps)
    a_perc = a_counts / (a_counts.sum()+eps)
    return float(np.sum((a_perc - e_perc) * np.log((a_perc+eps)/(e_perc+eps))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_data", required=True, help="CSV of new data with same columns as raw titanic")
    ap.add_argument("--threshold", type=float, default=0.2, help="PSI threshold to trigger retrain")
    args = ap.parse_args()

    with open("artifacts/train_feature_stats.json") as f:
        stats = json.load(f)

    df_new = pd.read_csv(args.new_data)

    # Map categorical to simple index for PSI
    numeric_cols = [c for c in stats if "value_counts" not in stats[c]]
    cat_cols = [c for c in stats if "value_counts" in stats[c]]

    psis = {}
    for c in numeric_cols:
        if c in df_new.columns:
            e = df_new[c].dropna()
            # build expected sample from mean/std (approx) if counts exist; else skip
            mu = stats[c].get("mean"); sd = stats[c].get("std") or 1.0
            if mu is None: continue
            np.random.seed(42)
            expected = np.random.normal(mu, sd, size=min(10000, max(1000, stats[c].get("n", 5000))))
            psis[c] = psi(expected, e.values)

    # Cat PSI: align distributions
    for c in cat_cols:
        if c in df_new.columns:
            base = stats[c]["value_counts"]
            e_keys = list(base.keys()); e_probs = np.array(list(base.values()), dtype=float)
            e_probs = e_probs / e_probs.sum()
            a_counts = df_new[c].value_counts().reindex(e_keys).fillna(0).values.astype(float)
            a_probs = a_counts / max(1.0, a_counts.sum())
            psis[c] = float(np.sum((a_probs - e_probs) * np.log((a_probs+1e-6)/(e_probs+1e-6))))

    mean_psi = float(np.mean(list(psis.values()))) if psis else 0.0
    print("Feature PSIs:", psis)
    print("Mean PSI:", mean_psi)

    if mean_psi > args.threshold:
        print(f"PSI {mean_psi:.3f} > threshold {args.threshold:.3f} → Retraining recommended.")
        os.system("dvc repro")
    else:
        print(f"PSI {mean_psi:.3f} ≤ threshold {args.threshold:.3f} → No retrain.")

if __name__ == "__main__":
    main()
