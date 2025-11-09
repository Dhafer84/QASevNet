import argparse, os, json, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from src.utils import LABELS, label2id, ensure_dir

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
    q = probs * bias[np.newaxis, :]
    q = q / q.sum(axis=1, keepdims=True)
    return q

def decide(p, maj_id, crit_id, maj_thresh, crit_thresh, delta):
    if p[crit_id] >= crit_thresh: return crit_id
    if p[maj_id]  >= maj_thresh:  return maj_id
    top2 = np.argsort(-p)[:2]
    if (maj_id in top2) and (p[top2[0]] - p[top2[1]] <= delta):
        return maj_id
    return int(np.argmax(p))

def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, interpolation='nearest'); ax.set_title('Matrice de confusion')
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_ylabel('Vrai'); ax.set_xlabel('Prédit')
    plt.tight_layout(); ensure_dir(os.path.dirname(path) + "/dummy"); plt.savefig(path); plt.close(fig)

def plot_precision_recall(y_true, probs, path):
    y_true_bin = np.zeros_like(probs)
    for i, y in enumerate(y_true): y_true_bin[i, y] = 1
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), probs.ravel())
    ap = average_precision_score(y_true_bin, probs, average='micro')
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, lw=2)
    ax.set_title(f'Courbe PR (micro) – AP={ap:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    plt.tight_layout(); ensure_dir(os.path.dirname(path) + "/dummy"); plt.savefig(path); plt.close(fig)

def main(args):
    df = pd.read_csv(args.test_csv)
    x = df["text"].astype(str).values
    y = df["label"].map(label2id).values

    probs = predict_with_export(x)

    # charge post-traitement si disponible
    pp_path = "models/postproc.json"
    if os.path.exists(pp_path):
        with open(pp_path, "r") as f:
            pp = json.load(f)
        maj_id = label2id["Majeur"]; crit_id = label2id["Critique"]
        bias = np.ones(len(LABELS), dtype=np.float32)
        bias[maj_id]  = pp.get("maj_bias", 1.0)
        bias[crit_id] = pp.get("crit_bias", 1.0)
        probs_adj = apply_bias(probs, bias)
        y_pred = np.array([
            decide(r, maj_id, crit_id,
                   pp.get("maj_thresh", 0.28),
                   pp.get("crit_thresh", 0.28),
                   pp.get("delta", 0.12))
            for r in probs_adj
        ])
    else:
        probs_adj = probs
        y_pred = probs_adj.argmax(axis=1)

    rep = classification_report(y, y_pred, target_names=LABELS, digits=3, zero_division=0)
    ensure_dir("reports/dummy")
    with open("reports/classification_report.txt", "w") as f: f.write(rep)
    print(rep)

    plot_confusion(y, y_pred, args.cm_path)
    plot_precision_recall(y, probs_adj, args.pr_path)
    print(f"Saved: {args.cm_path} & {args.pr_path} & reports/classification_report.txt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default="data/test.csv")
    p.add_argument("--model_dir", default="models/qasevnet.keras")
    p.add_argument("--cm_path", default="reports/confusion_matrix.png")
    p.add_argument("--pr_path", default="reports/precision_recall.png")
    main(p.parse_args())
