# BikeNow MLOps

Projet pédagogique MLOps autour du dataset **Bike Sharing Dataset**.

Objectif métier : prédire la demande horaire de vélos pour l'entreprise fictive **BikeNow**.

Ce dépôt sert de base pour les TPs :

1. Premier pipeline ML/MLOps
2. Qualité des données, EDA et split temporel
3. Feature engineering, baselines et modèles
4. Pipeline sklearn reproductible et évaluation avancée
5. Tracking des expériences avec MLflow
6. Registry simplifié et validation automatique
7. Déploiement batch/API et monitoring simple
8. Projet final

---

## 1. Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Données

Placer le fichier `hour.csv` ici :

```text
data/raw/hour.csv
```

Le fichier vient du **Bike Sharing Dataset**.

---

## 3. Commandes principales

Contrôle qualité :

```bash
python src/data_quality.py
```

Préparation train/test :

```bash
python src/prepare_data.py
```

Premier entraînement :

```bash
python src/train_first_model.py
```

Pipeline complet :

```bash
python src/train_pipeline.py
```

Prédiction batch :

```bash
python src/predict_batch.py
```

API FastAPI :

```bash
uvicorn src.api:app --reload
```

Tests :

```bash
pytest
```

---

## 4. Structure

```text
bikenow-mlops/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── input/
│   └── output/
├── logs/
├── models/
├── notebooks/
├── registry/
│   ├── candidate/
│   └── production/
├── reports/
│   └── figures/
├── runs/
├── src/
├── tests/
├── .env.example
├── .gitignore
├── config.yaml
├── Makefile
├── README.md
└── requirements.txt
```

---

## 5. Principe MLOps

Le but n'est pas uniquement d'entraîner un modèle.

Le projet doit produire des artefacts :

- modèles sauvegardés ;
- métriques ;
- rapports ;
- logs ;
- runs ;
- registry simplifié ;
- prédictions ;
- preuves de validation.
