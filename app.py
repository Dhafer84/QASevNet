# app.py
import os, json, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# pour importer src.utils quand app.py est à la racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.utils import LABELS, label2id, id2label, load_pickle, top_terms_from_tfidf

st.set_page_config(page_title="QASevNet", page_icon="🛠️", layout="centered")
# ---------- Affichage du logo et du titre ----------
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=90)
with col2:
    st.title("QASevNet — Classification de la criticité")
    st.caption("🧠 Démo : description → criticité (Mineur / Majeur / Critique) avec calibration post-traitement")

# ------------------------ Utils ------------------------

def softmax_with_temperature(p: np.ndarray, T: float = 1.0) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    z = p / max(T, 1e-6)
    z = z - np.max(z)                    # stabilité numérique
    ez = np.exp(z)
    s = ez.sum()
    return (ez / s) if s > 0 else p

@st.cache_resource
def load_assets():
    saved = tf.saved_model.load("models/qasevnet_export")
    infer = saved.signatures.get("serve") or saved.signatures[list(saved.signatures.keys())[0]]
    tfidf = load_pickle("models/tfidf.pkl")
    return infer, tfidf

def load_postproc():
    pp_path = "models/postproc.json"
    if os.path.exists(pp_path):
        with open(pp_path, "r") as f:
            return json.load(f)
    # défauts raisonnables
    return {"maj_bias": 1.0, "crit_bias": 1.0, "maj_thresh": 0.28, "crit_thresh": 0.28, "delta": 0.12}

def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    q = probs * bias
    s = q.sum()
    if s > 0: q = q / s
    return q

def decide_with_pp(p: np.ndarray, pp: dict) -> int:
    maj_id, crit_id = label2id["Majeur"], label2id["Critique"]
    if p[crit_id] >= pp.get("crit_thresh", 0.28): return crit_id
    if p[maj_id]  >= pp.get("maj_thresh", 0.28): return maj_id
    top2 = np.argsort(-p)[:2]
    if (maj_id in top2) and (p[top2[0]] - p[top2[1]] <= pp.get("delta", 0.12)): return maj_id
    return int(np.argmax(p))

def infer_probs(infer, text: str) -> np.ndarray:
    """
    Appel robuste de la signature SavedModel exportée par Keras 3.
    Essaie plusieurs noms d'entrée et formes (1D et 2D).
    Retourne un vecteur de proba de taille len(LABELS).
    """
    t1 = tf.constant([text], dtype=tf.string)                                 # [N]
    t2 = tf.constant(np.array([text]).reshape(-1, 1), dtype=tf.string)        # [N,1]
    common_keys = ("text", "inputs", "input_1", "args_0")

    for tensor in (t1, t2):
        # 1) clés standard
        for kw in common_keys:
            try:
                out = infer(**{kw: tensor})
                return list(out.values())[0].numpy()[0]
            except Exception:
                pass
        # 2) clé exportée (dynamique)
        try:
            key = list(infer.structured_input_signature[1].keys())[0]
            out = infer(**{key: tensor})
            return list(out.values())[0].numpy()[0]
        except Exception:
            pass

    raise RuntimeError("Impossible d'appeler la signature d'inférence (forme/nom d'entrée).")

infer, tfidf = load_assets()
# ---- Helpers pour l'évaluation à la volée ----
import io, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

def infer_probs_texts(infer, texts):
    """Retourne un np.array (N, C) de probabilités pour une liste de textes."""
    import numpy as np, tensorflow as tf
    t = tf.constant(np.array(texts, dtype=object).reshape(-1, 1), dtype=tf.string)
    try:
        res = infer(text=t)                 # signature standard
    except Exception:
        try:
            res = infer(args_0=t)           # signature Keras 3
        except Exception:
            key = list(infer.structured_input_signature[1].keys())[0]
            res = infer(**{key: t})         # autre nom
    return list(res.values())[0].numpy()

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion")
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color=("black" if cm[i,j] < cm.max()/2 else "white"))
    ax.set_ylabel("Vrai"); ax.set_xlabel("Prédit")
    fig.tight_layout()
    return fig

def plot_pr_curve_micro(y_true_ids, probs):
    import numpy as np
    y_true_bin = np.zeros_like(probs)
    for i, y in enumerate(y_true_ids):
        y_true_bin[i, y] = 1
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), probs.ravel())
    ap = average_precision_score(y_true_bin, probs, average='micro')
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, lw=2)
    ax.set_title(f"Courbe PR (micro) – AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    fig.tight_layout()
    return fig

# ------------------------ UI ------------------------

#st.title("QASevNet — Classification de la criticité")
#st.caption("Démo : description → criticité (Mineur / Majeur / Critique) avec calibration post-traitement")

tab_pred, tab_eval = st.tabs(["🔮 Prédiction", "📊 Évaluation"])

with tab_pred:
    with st.form("predict_form", clear_on_submit=False):
        text = st.text_area(
            "Décrivez le bug (texte libre)",
            height=160,
            placeholder="Ex: La vérification 2FA n’est pas demandée sur certains parcours…"
        )
        colA, colB = st.columns([1,1])
        with colA:
            T = st.slider("Température (lissage des probabilités)", 0.5, 2.0, 1.0, 0.05)
        with colB:
            show_raw = st.checkbox("Afficher aussi les probabilités brutes", value=False)
        submitted = st.form_submit_button("Prédire")

    if submitted and text.strip():
        try:
            # 1) Inférence modèle exporté (robuste)
            probs = infer_probs(infer, text).astype("float32")

            # 2) Lissage + post-traitement calibré
            probs_T = softmax_with_temperature(probs, T=T)
            pp = load_postproc()
            bias = np.ones(len(LABELS), dtype=np.float32)
            bias[label2id["Majeur"]]   = pp.get("maj_bias", 1.0)
            bias[label2id["Critique"]] = pp.get("crit_bias", 1.0)
            probs_adj = apply_bias(probs_T, bias)

            pred_id = decide_with_pp(probs_adj, pp)
            pred = LABELS[pred_id]

            # 3) Affichage — utiliser un DataFrame indexé (compatibilité Streamlit)
            st.subheader(f"Prédiction : **{pred}**")
            st.markdown("**Probabilités (après calibration)**")
            df_probs = pd.DataFrame(
                {"Classe": LABELS, "Probabilité": [float(probs_adj[i]) for i in range(len(LABELS))]}
            ).set_index("Classe")
            st.bar_chart(df_probs["Probabilité"])

            cols = st.columns(3)
            with cols[0]:
                st.metric("Seuil Majeur", f"{pp.get('maj_thresh',0.28):.2f}")
                st.metric("Biais Majeur", f"{pp.get('maj_bias',1.0):.2f}")
            with cols[1]:
                st.metric("Seuil Critique", f"{pp.get('crit_thresh',0.28):.2f}")
                st.metric("Biais Critique", f"{pp.get('crit_bias',1.0):.2f}")
            with cols[2]:
                st.metric("Δ top-2", f"{pp.get('delta',0.12):.2f}")

            if show_raw:
                st.divider()
                st.markdown("**Probabilités brutes (avant calibration/post-traitement)**")
                st.write({lbl: float(probs[i]) for i, lbl in enumerate(LABELS)})

            st.divider()
            st.markdown("#### Mots importants (TF-IDF)")
            try:
                topk = top_terms_from_tfidf(tfidf, text, k=8)
                if topk:
                    for ttoken, w in topk:
                        st.write(f"- {ttoken} · {w:.3f}")
                else:
                    st.write("Pas d'explication disponible pour ce texte.")
            except Exception:
                st.info("Explications TF-IDF indisponibles.")
        except Exception as e:
            st.error(f"Erreur d'inférence : {e}")

with tab_eval:
    st.markdown("Cet onglet affiche les artefacts d’évaluation.")

    # 1) Si des fichiers existent (poussés via Option A), on les montre
    cm_path = "reports/confusion_matrix.png"
    pr_path = "reports/precision_recall.png"
    cr_txt  = "reports/classification_report.txt"

    files_found = os.path.exists(cm_path) and os.path.exists(pr_path) and os.path.exists(cr_txt)

    if files_found:
        cols = st.columns(2)
        with cols[0]:
            st.image(cm_path, caption="Matrice de confusion")
        with cols[1]:
            st.image(pr_path, caption="Precision/Recall (micro)")
        with open(cr_txt, "r", encoding="utf-8", errors="ignore") as f:
            st.markdown("---")
            st.code(f.read())
    else:
        # 2) Sinon, on propose de générer à la volée
        st.info("Aucun artefact trouvé dans `reports/`. Cliquez pour calculer les métriques à partir de `data/test.csv`.")
        gen = st.button("⚙️ Générer l’évaluation maintenant")
        if gen:
            try:
                df = pd.read_csv("data/test.csv")
                x = df["text"].astype(str).tolist()
                y = df["label"].map(label2id).astype(int).values

                probs = infer_probs_texts(infer, x)
                y_pred = probs.argmax(axis=1)

                # Rapports
                rep = classification_report(y, y_pred, target_names=LABELS, digits=3, zero_division=0)
                st.markdown("### Rapport de classification")
                st.code(rep)

                # Graphes inline
                cm = confusion_matrix(y, y_pred, labels=list(range(len(LABELS))))
                fig_cm = plot_confusion(cm)
                st.pyplot(fig_cm)

                fig_pr = plot_pr_curve_micro(y, probs)
                st.pyplot(fig_pr)

                # Option : sauvegarder aussi dans /reports pour usage futur
                os.makedirs("reports", exist_ok=True)
                with open("reports/classification_report.txt", "w") as f:
                    f.write(rep)
                fig_cm.savefig("reports/confusion_matrix.png"); plt.close(fig_cm)
                fig_pr.savefig("reports/precision_recall.png"); plt.close(fig_pr)
                st.success("Évaluation générée et sauvegardée dans `reports/`.")

            except Exception as e:
                st.error(f"Échec de l’évaluation : {e}")


st.sidebar.header("À propos")
st.sidebar.write("Modèle: TextVectorization(n-grams TF-IDF) → (Dense 128) → Softmax.")
st.sidebar.write("Calibration via `models/postproc.json` (biais & seuils).")
st.sidebar.markdown("**Utilisation**\n1) Entrer une description.\n2) Régler (optionnel) la température.\n3) Cliquer *Prédire*.\n4) Voir *📊 Évaluation* pour les métriques.")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Dhafer-QASevNet")
