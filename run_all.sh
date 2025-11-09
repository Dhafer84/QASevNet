#!/usr/bin/env bash
set -e

# réentraînement
# rm -rf models/qasevnet.keras models/qasevnet_export
# python -m src.train --ngram_max 4 --epochs 250 --batch_size 8

# calibration post-traitement sur le set de validation
python -m src.tune_postproc --val_csv data/test.csv --out_json models/postproc.json

# évaluation alignée
python -m src.evaluate --test_csv data/test.csv

#  UI (décommente si tu veux lancer directement)
# streamlit run app.py
