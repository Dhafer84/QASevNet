import argparse, json, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import f1_score, classification_report
from src.utils import LABELS, label2id

def predict_with_export(x_texts):
    saved = tf.saved_model.load("models/qasevnet_export")
    infer = saved.signatures.get("serve") or saved.signatures[list(saved.signatures.keys())[0]]
    x = np.array(x_texts).reshape(-1, 1).astype(object)
    t = tf.constant(x, dtype=tf.string)
    try:    res = infer(t)
    except TypeError: res = infer(args_0=t)
    probs = next(iter(res.values())).numpy()
    return probs

def apply_bias(probs, bias):
    """Multiplie les proba par des biais par classe, puis renormalise."""
    q = probs * bias[np.newaxis, :]
    q = q / q.sum(axis=1, keepdims=True)
    return q

def decide(probs_row, maj_id, crit_id, maj_thresh, crit_thresh, delta):
    # seuils par classe
    if probs_row[crit_id] >= crit_thresh:
        return crit_id
    if probs_row[maj_id] >= maj_thresh:
        return maj_id
    # top-2 + delta
    top2 = np.argsort(-probs_row)[:2]
    if (maj_id in top2) and (probs_row[top2[0]] - probs_row[top2[1]] <= delta):
        return maj_id
    return int(np.argmax(probs_row))

def grid_search(y_true, probs, maj_id, crit_id):
    # Grilles compactes mais efficaces (ajuste si besoin)
    maj_bias_grid  = [0.9, 1.0, 1.1, 1.2, 1.35, 1.5]
    crit_bias_grid = [0.9, 1.0, 1.1, 1.2, 1.35]
    maj_th_grid    = [0.20, 0.24, 0.28, 0.32, 0.36]
    crit_th_grid   = [0.20, 0.24, 0.28, 0.32, 0.36]
    delta_grid     = [0.08, 0.12, 0.16, 0.20]

    best = {"f1": -1, "maj_bias":1.0, "crit_bias":1.0, "maj_thresh":0.28, "crit_thresh":0.28, "delta":0.12}
    for mb in maj_bias_grid:
        for cb in crit_bias_grid:
            bias = np.ones(len(LABELS), dtype=np.float32)
            bias[maj_id]  = mb
            bias[crit_id] = cb
            q = apply_bias(probs, bias)
            for mt in maj_th_grid:
                for ct in crit_th_grid:
                    for d in delta_grid:
                        y_hat = np.array([decide(r, maj_id, crit_id, mt, ct, d) for r in q])
                        f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
                        if f1 > best["f1"]:
                            best = {"f1": float(f1), "maj_bias":float(mb), "crit_bias":float(cb),
                                    "maj_thresh":float(mt), "crit_thresh":float(ct), "delta":float(d)}
    return best

def main(args):
    df = pd.read_csv(args.val_csv)
    x = df["text"].astype(str).values
    y = df["label"].map(label2id).values
    probs = predict_with_export(x)

    maj = label2id["Majeur"]; crit = label2id["Critique"]
    best = grid_search(y, probs, maj, crit)

    # Sauvegarde
    best["labels"] = LABELS
    with open(args.out_json, "w") as f:
        json.dump(best, f, indent=2)
    print("Best postproc:", best)

    # Affiche un rapport avec ces paramètres
    bias = np.ones(len(LABELS), dtype=np.float32)
    bias[maj]  = best["maj_bias"]
    bias[crit] = best["crit_bias"]
    q = apply_bias(probs, bias)
    y_hat = np.array([decide(r, maj, crit, best["maj_thresh"], best["crit_thresh"], best["delta"]) for r in q])
    print(classification_report(y, y_hat, target_names=LABELS, digits=3, zero_division=0))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default="data/test.csv")  # utilise ton set de validation
    p.add_argument("--out_json", default="models/postproc.json")
    main(p.parse_args())
