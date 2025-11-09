import json, os, pickle
import numpy as np
from typing import List, Tuple

LABELS = ["Mineur","Majeur","Critique"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def top_terms_from_tfidf(vectorizer, text: str, k: int = 8) -> List[Tuple[str, float]]:
    """Return top-k terms by tf-idf weight in this single text."""
    X = vectorizer.transform([text])
    if X.shape[1] != len(vectorizer.vocabulary_):
        # nothing
        return []
    row = X.tocsr()
    data = row.data
    indices = row.indices
    pairs = [(vectorizer.get_feature_names_out()[idx], float(w)) for idx, w in zip(indices, data)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]
