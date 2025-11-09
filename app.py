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
    st.markdown("Cet onglet affiche les artefacts d’évaluation générés par `src.evaluate`.")
    cm_path = "reports/confusion_matrix.png"
    pr_path = "reports/precision_recall.png"
    cr_txt  = "reports/classification_report.txt"

    cols = st.columns(2)
    with cols[0]:
        if os.path.exists(cm_path):
            # Anciennes versions de Streamlit : pas de use_container_width
            st.image(cm_path, caption="Matrice de confusion")
        else:
            st.info("`reports/confusion_matrix.png` introuvable. Lance `python -m src.evaluate ...`")
    with cols[1]:
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision/Recall (micro)")
        else:
            st.info("`reports/precision_recall.png` introuvable.")

    st.markdown("---")
    if os.path.exists(cr_txt):
        with open(cr_txt, "r", encoding="utf-8", errors="ignore") as f:
            st.code(f.read())
    else:
        st.info("`reports/classification_report.txt` introuvable.")

st.sidebar.header("À propos")
st.sidebar.write("Modèle: TextVectorization(n-grams TF-IDF) → (Dense 128) → Softmax.")
st.sidebar.write("Calibration via `models/postproc.json` (biais & seuils).")
st.sidebar.markdown("**Utilisation**\n1) Entrer une description.\n2) Régler (optionnel) la température.\n3) Cliquer *Prédire*.\n4) Voir *📊 Évaluation* pour les métriques.")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Dhafer-QASevNet")
