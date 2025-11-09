# 🧠 **QASevNet – Classification automatique de la criticité des anomalies**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dhafer84-qasevnet-app-tckpoi.streamlit.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)]() [![Python](https://img.shields.io/badge/Python-3.11-blue)]() [![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)]()

---

## 🚀 **Aperçu du projet**

**QASevNet** est une application web IA basée sur **TensorFlow + Streamlit**, permettant de **classifier automatiquement la criticité d’une anomalie logicielle** selon sa description textuelle :

> **Criticités :**
>
> * 🟢 *Mineur*
> * 🟡 *Majeur*
> * 🔴 *Critique*

Elle combine un modèle entraîné avec `TextVectorization (TF-IDF)` et une **calibration post-traitée** pour obtenir des résultats cohérents, même sur des petits datasets.

---

## 🎯 **Objectifs**

* Automatiser la catégorisation de bugs dans un contexte QA / Qualité Logicielle
* Faciliter la priorisation des anomalies (Mineur / Majeur / Critique)
* Offrir une démo publique simple à utiliser via **Streamlit Cloud**

---

## 🧩 **Architecture globale**

```mermaid
graph TD
A[Description du bug 📝] --> B[Vectorisation TF-IDF]
B --> C[Modèle Dense(128) + Softmax]
C --> D[Calibration post-proc JSON]
D --> E[Prédiction Criticité]
```

### Stack Technique :

* **Python 3.11**
* **TensorFlow 2.17 (CPU)**
* **Scikit-learn** pour TF-IDF + métriques
* **Streamlit 1.31** pour l’interface web
* **Matplotlib** pour les visualisations (Matrice de confusion & PR Curve)

---

## 🧠 **Fonctionnalités principales**

| Fonction                     | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| 🔮 **Prédiction**            | Entrez une description d’anomalie → obtention de la criticité prédite |
| 🧾 **Explications TF-IDF**   | Affiche les mots-clés les plus influents                              |
| 📊 **Évaluation intégrée**   | Matrice de confusion + courbe Precision/Recall                        |
| 🧰 **Calibration dynamique** | Ajustement par biais & seuils via `postproc.json`                     |
| 📦 **Modèle exporté**        | Modèle TensorFlow sauvegardé sous `models/qasevnet_export/`           |

---

## 🧪 **Exemple d’utilisation**

**Exemple de texte :**

```
L’application plante lors de la génération du PDF si le fichier dépasse 10 Mo.
```

**Résultat :**

```
Prédiction : Critique
Probabilités : {'Mineur': 0.05, 'Majeur': 0.12, 'Critique': 0.83}
```

---

## 📈 **Évaluation du modèle**

| Classe               | Précision | Rappel | F1-score |
| -------------------- | --------- | ------ | -------- |
| Mineur               | 0.50      | 0.20   | 0.29     |
| Majeur               | 0.50      | 0.75   | 0.60     |
| Critique             | 0.83      | 1.00   | 0.91     |
| **Accuracy globale** | **0.64**  |        |          |

📊 Visualisations :

* `reports/confusion_matrix.png`
* `reports/precision_recall.png`

---

## ⚙️ **Installation locale**

```bash
# 1️⃣ Cloner le repo
git clone https://github.com/Dhafer84/QASevNet.git
cd QASevNet

# 2️⃣ Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate   # (sous mac/linux)
# ou .venv\Scripts\activate  # (sous Windows)

# 3️⃣ Installer les dépendances
pip install -r requirements.txt

# 4️⃣ Lancer Streamlit
streamlit run app.py
```

---

## ☁️ **Démo en ligne**

🔗 [Accéder à l’application QASevNet sur Streamlit Cloud](https://dhafer84-qasevnet-app-tckpoi.streamlit.app)

---

## 📁 **Structure du projet**

```
QASevNet/
│
├── app.py                  # Application principale Streamlit
├── src/                    # Scripts du modèle et de l’évaluation
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── tune_postproc.py
│
├── data/                   # Jeux de données
│   └── test.csv
│
├── models/                 # Modèle TensorFlow exporté
│   ├── qasevnet.keras
│   ├── qasevnet_export/
│   └── postproc.json
│
├── reports/                # Visualisations et rapports
│   ├── confusion_matrix.png
│   ├── precision_recall.png
│   └── classification_report.txt
│
└── requirements.txt        # Dépendances Python
```

---

## 🧩 **Auteur**

👤 **Dhafer Bouthelja**
💼 Ingénieur Qualité Logicielle & DevOps
🔗 [LinkedIn](https://www.linkedin.com/in/bouthelja-dhafer-116681a0/)
📧 Contact : *[dhafer.bouthelja@gmail.com](mailto:dhafer.bouthelja@gmail.com)* (ou via LinkedIn)

---

## 🌟 **Remerciements**

* TensorFlow & Streamlit pour leurs écosystèmes open-source
* Communauté IA tunisienne 🇹🇳 pour le partage et la passion ❤️

---
