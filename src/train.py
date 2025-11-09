import argparse, os, random, csv, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
from collections import Counter

from src.utils import LABELS, label2id, ensure_dir, save_pickle

# Reproductibilité
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------- FOCAL LOSS (compat TF 2.16 / Keras 3) ----------
def sparse_categorical_focal_loss(gamma=2.5, alpha=None, from_logits=False):
    """
    Focal Loss pour labels entiers (sparse).
    alpha: None OU liste/np.array de poids par classe, p.ex. [1.0, 2.2, 1.2]
    """
    alpha = None if alpha is None else tf.convert_to_tensor(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        if from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        depth = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=depth)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        if alpha is not None:
            alpha_t = tf.gather(alpha, y_true)
            alpha_t = tf.squeeze(alpha_t, axis=-1) if alpha_t.shape.ndims == 2 else alpha_t
        else:
            alpha_t = 1.0
        focal = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)
    return loss_fn
# -----------------------------------------------------------

def compute_class_weight(y_ids):
    counts = Counter(y_ids.tolist())
    total = sum(counts.values())
    cw = {}
    for i in range(len(LABELS)):
        freq = counts.get(i, 1)
        cw[i] = total / (len(LABELS) * freq)
    return cw

class MacroF1Callback(Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.best_f1 = -1.0
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.x_val, verbose=0)
        y_hat = y_prob.argmax(axis=1)
        f1_macro = f1_score(self.y_val, y_hat, average="macro", zero_division=0)
        if logs is not None:
            logs["val_macro_f1"] = f1_macro
        if f1_macro > self.best_f1:
            self.best_f1 = f1_macro
            self.best_epoch = epoch
        print(f" — val_macro_f1: {f1_macro:.4f}")

def main(args):
    # --- 1) Données ---
    train = pd.read_csv(args.train_csv)
    val   = pd.read_csv(args.val_csv)

    x_train = train["text"].astype(str).values
    y_train = train["label"].map(label2id).values
    x_val   = val["text"].astype(str).values
    y_val   = val["label"].map(label2id).values

    # --- 2) TextVectorization n-grams (TF-IDF) ---
    text_vec = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        max_tokens=60000,
        ngrams=args.ngram_max,          # essaye 3 à 4
        output_mode="tf_idf",
        name="text_vectorization",
    )
    ds = tf.data.Dataset.from_tensor_slices(x_train).batch(64)
    with tf.device("/CPU:0"):
        text_vec.adapt(ds)

    # --- Priors pour initialiser le biais de la couche softmax ---
    class_counts = np.bincount(y_train, minlength=len(LABELS))
    priors = class_counts / class_counts.sum()
    bias_init = tf.keras.initializers.Constant(np.log(priors + 1e-8))

    # ---  Modèle : TF-IDF -> Dropout -> Dense(128) -> Dropout -> Softmax ---
    text_in = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = text_vec(text_in)
    x = layers.Dropout(0.12)(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(max(args.l2, 8e-7))
    )(x)
    x = layers.Dropout(0.18)(x)
    out = layers.Dense(
        len(LABELS),
        activation="softmax",
        name="classifier",
        bias_initializer=bias_init,
    )(x)
    model = models.Model(text_in, out)

    # --- Compilation : Focal Loss avec fort alpha sur Majeur ---
    alpha = [1.0] * len(LABELS)
    alpha[label2id["Majeur"]]   = 1.4   # au lieu de 1.8 / 2.2
    alpha[label2id["Critique"]] = 1.15  # un petit plus, sans écraser Mineur
    loss = sparse_categorical_focal_loss(gamma=2.2, alpha=alpha, from_logits=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
                  loss=loss, metrics=["accuracy"])

    # --- Entraînement ---
    class_weight = compute_class_weight(y_train)  # dataset équilibré → proche de 1
    print("Class weight used:", class_weight)

    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_macro_f1", factor=0.5, patience=6, min_lr=1e-5, verbose=1, mode="max"
    )
    es = tf.keras.callbacks.EarlyStopping(
        patience=12, monitor="val_macro_f1", mode="max", restore_best_weights=True
    )
    f1cb = MacroF1Callback(x_val, y_val)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=2,
        callbacks=[es, rlrop, f1cb],
    )

    # --- Sauvegardes (.keras + SavedModel) ---
    parent_dir = os.path.dirname(args.model_dir) or "."
    ensure_dir(os.path.join(parent_dir, "dummy"))
    model.save(args.model_dir)
    print(f"Saved Keras model to {args.model_dir}")

    export_dir = "models/qasevnet_export"
    ensure_dir(export_dir + "/dummy")
    model.export(export_dir)  # Keras 3 : export inclut TextVectorization
    print(f"Exported inference model to {export_dir}")

    # --- TF-IDF scikit-learn (explications UI) ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf.fit(pd.concat([train["text"], val["text"]]).astype(str).values)
    save_pickle(tfidf, args.tfidf_path)

    # ---  Log minimal pour tracer les essais ---
    os.makedirs("reports", exist_ok=True)
    log_path = "reports/train_log.csv"
    val_acc = float(model.evaluate(x_val, y_val, verbose=0)[1])
    val_macro_f1 = getattr(f1cb, "best_f1", None)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["ts","ngram","l2","epochs","batch","val_acc","val_macro_f1","weights","alpha","gamma","lr"])
        w.writerow([time.strftime("%F %T"), args.ngram_max, args.l2, args.epochs,
                    args.batch_size, val_acc, val_macro_f1, json.dumps(class_weight),
                    json.dumps(alpha), 2.5, 8e-4])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv",   default="data/test.csv")
    parser.add_argument("--model_dir", default="models/qasevnet.keras")
    parser.add_argument("--tfidf_path", default="models/tfidf.pkl")
    parser.add_argument("--ngram_max", type=int, default=4)
    parser.add_argument("--l2",        type=float, default=8e-7)
    parser.add_argument("--epochs",    type=int,   default=250)
    parser.add_argument("--batch_size",type=int,   default=8)
    args = parser.parse_args()
    main(args)
