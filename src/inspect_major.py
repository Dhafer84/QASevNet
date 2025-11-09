import pandas as pd, numpy as np, argparse
from src.utils import label2id
from src.evaluate import predict_with_export

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/test.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    x = df["text"].astype(str).values
    y = df["label"].map(label2id).values
    probs = predict_with_export(x)
    y_hat = probs.argmax(1)

    maj = label2id["Majeur"]
    idx_err = np.where((y == maj) & (y_hat != maj))[0]

    print(f"Erreurs Majeur: {len(idx_err)}")
    for i in idx_err[:10]:
        p = probs[i]
        top2 = np.argsort(-p)[:2]
        print("-"*60)
        print("Texte:", x[i])
        print("Probas:", {k: float(p[j]) for j,k in enumerate(['Mineur','Majeur','Critique'])})
        print("Top2:", top2, "Décision top1:", int(p.argmax()))
