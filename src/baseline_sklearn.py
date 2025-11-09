import pandas as pd, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import LABELS, label2id

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

def main(args):
    train = pd.read_csv(args.train_csv)
    test  = pd.read_csv(args.test_csv)

    X_tr, y_tr = train["text"].astype(str).values, train["label"].map(label2id).values
    X_te, y_te = test["text"].astype(str).values,  test["label"].map(label2id).values

    word_tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=1)
    char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1, max_features=80000)

    # Union (mots + caractères) 
    from scipy.sparse import hstack
    Xw = word_tfidf.fit_transform(X_tr)
    Xc = char_tfidf.fit_transform(X_tr)
    X_train = hstack([Xw, Xc])

    Xw_te = word_tfidf.transform(X_te)
    Xc_te = char_tfidf.transform(X_te)
    X_test = hstack([Xw_te, Xc_te])

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_tr)
    y_pred = clf.predict(X_test)

    print(classification_report(y_te, y_pred, target_names=LABELS, digits=3, zero_division=0))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--test_csv",  default="data/test.csv")
    main(p.parse_args())
