import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
import numpy as np
import os, json, numpy as np
from src.utils import LABELS, label2id

from src.utils import LABELS, id2label, load_pickle, top_terms_from_tfidf, label2id

st.set_page_config(page_title="QASevNet", page_icon="🛠️", layout="centered")

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
    if s > 0:
        q = q / s
    return q

def decide_with_pp(p: np.ndarray, pp: dict) -> int:
    maj_id, crit_id = label2id["Majeur"], label2id["Critique"]
    if p[crit_id] >= pp.get("crit_thresh", 0.28):
        return crit_id
    if p[maj_id] >= pp.get("maj_thresh", 0.28):
        return maj_id
    top2 = np.argsort(-p)[:2]
    if (maj_id in top2) and (p[top2[0]] - p[top2[1]] <= pp.get("delta", 0.12)):
        return maj_id
    return int(np.argmax(p))

def apply_keyword_prior(text: str, probs: np.ndarray) -> np.ndarray:
    tl = text.lower()
    boost = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mineur = ["orthographe","typo","alignement","espacement","icone","icône",
              "placeholder","couleur","traduction","contraste","padding","margin"]
    majeur = ["2fa","double facteur","auth","authentification","token","jeton",
              "session","déconnexion","paiement","api","timeout","latence",
              "upload","download","webhook","filtre","tri","cache","retry","time-out","queue"]
    critique = ["crash","plante","perte de données","corruption","bloqué","blocage",
                "fuite","leak","élévation","elevation","admin","root","sql injection","xss","csrf","rce","deadlock"]

    if any(w in tl for w in mineur):   boost[0] *= 1.25
    if any(w in tl for w in majeur):   boost[1] *= 2.00
    if any(w in tl for w in critique): boost[2] *= 1.10

    probs = probs * boost
    s = probs.sum()
    if s > 0: probs = probs / s
    return probs

def softmax_with_temperature(p, T: float = 1.0):
    p = np.asarray(p, dtype=np.float32)
    z = np.log(np.clip(p, 1e-9, 1.0))
    z = z / max(T, 1e-6)
    z = z - z.max()
    ez = np.exp(z)
    return ez / ez.sum()

def decide_with_top2_and_threshold(p: np.ndarray, maj_id: int, delta: float = 0.18, maj_thresh: float = 0.22):
    if p[maj_id] >= maj_thresh:
        return maj_id
    top2 = np.argsort(-p)[:2]
    if (maj_id in top2) and (p[top2[0]] - p[top2[1]] <= delta):
        return maj_id
    return int(np.argmax(p))

st.title("QASevNet — Classification de la criticité des anomalies")
st.caption("Démo : description → criticité (Mineur / Majeur / Critique) + explications TF-IDF + prior mots-clés")

infer, tfidf = load_assets()

tab_pred, tab_eval = st.tabs(["🔮 Prédiction", "📊 Évaluation"])

with tab_pred:
    with st.form("predict_form"):
        text = st.text_area(
            "Décrivez le bug (texte libre)",
            height=160,
            placeholder="Ex: L'application crash au paiement si le montant dépasse 10 000..."
        )
        T = st.slider("Température (lissage des probabilités)", 0.5, 2.0, 1.0, 0.05)
        submitted = st.form_submit_button("Prédire")
    if submitted and text.strip():
        # Inférence modèle exporté (SavedModel)
        res = infer(text=tf.constant([text]))
        probs = list(res.values())[0].numpy()[0].astype("float32")
        T = st.slider("Température (lissage)", 0.5, 2.0, 1.0, 0.05)
        probs_T = softmax_with_temperature(probs, T=T)
        pp = load_postproc()
        bias = np.ones(len(LABELS), dtype=np.float32)
        bias[label2id["Majeur"]]  = pp.get("maj_bias", 1.0)
        bias[label2id["Critique"]] = pp.get("crit_bias", 1.0)
        probs_adj = apply_bias(probs_T, bias)
        pred_id = decide_with_pp(probs_adj, pp)
        pred = LABELS[pred_id]

        st.subheader(f"Prédiction : **{pred}**")
        st.markdown("**Probabilités (après température & prior mots-clés)**")
        prob_map = {lbl: float(probs_adj[i]) for i, lbl in enumerate(LABELS)}
        st.write(prob_map)
        st.bar_chart({"Probabilité": [prob_map[lbl] for lbl in LABELS]}, x=LABELS)

        # Alerte faible confiance si l’écart top1-top2 est petit
        top2 = np.argsort(-probs_adj)[:2]
        margin = float(probs_adj[top2[0]] - probs_adj[top2[1]])
        if margin < 0.05:
            st.warning(
                f"Confiance faible (écart top-1/top-2 = {margin:.3f}). "
                "Revoir la description ou ajouter des indices contextuels."
            )

        st.divider()
        st.markdown("#### Mots importants (TF-IDF)")
        topk = top_terms_from_tfidf(tfidf, text, k=8)
        if topk:
            for t, w in topk:
                st.write(f"- {t} · {w:.3f}")
        else:
            st.write("Pas d'explication disponible pour ce texte.")

with tab_eval:
    st.markdown("Cet onglet affiche les artefacts d’évaluation générés par `src.evaluate`.")
    cm_path = "reports/confusion_matrix.png"
    pr_path = "reports/precision_recall.png"
    cr_txt  = "reports/classification_report.txt"

    cols = st.columns(2)
    with cols[0]:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Matrice de confusion", use_container_width=True)
        else:
            st.info("`reports/confusion_matrix.png` introuvable. Lance `python -m src.evaluate ...`")
    with cols[1]:
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision/Recall (micro)", use_container_width=True)
        else:
            st.info("`reports/precision_recall.png` introuvable.")

    st.markdown("---")
    if os.path.exists(cr_txt):
        with open(cr_txt, "r", encoding="utf-8", errors="ignore") as f:
            st.code(f.read())
    else:
        st.info("`reports/classification_report.txt` introuvable.")

st.sidebar.header("À propos")
st.sidebar.write("Modèle: TextVectorization(n-grams) → Dense(128, ReLU) → Dropout → Dense(3, Softmax).")
st.sidebar.write("Explications légères via TF-IDF. Prior mots-clés (léger) pour aider l’interprétation.")
st.sidebar.markdown("**Comment utiliser**\n1) Saisir une description de bug.\n2) Choisir la température.\n3) Cliquer *Prédire*.\n4) Voir *📊 Évaluation* pour les métriques.")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 QASevNet")
