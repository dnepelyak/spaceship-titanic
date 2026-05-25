# 🚀 Spaceship Titanic — Neural Network Ensemble

[![CI](https://github.com/YOUR_USERNAME/spaceship-titanic/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/spaceship-titanic/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-green)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

Binary classification — predict whether a passenger was transported to an alternate dimension.

---

## 🏗 Architecture

```
spaceship-titanic/
├── data/                    ← Place Kaggle CSVs here (not tracked by Git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── 01_EDA.ipynb         ← Exploratory Data Analysis
├── src/
│   ├── preprocessing.py     ← Feature engineering + sklearn pipeline
│   ├── model.py             ← MLP & Wide&Deep Keras models
│   └── train.py             ← Main: HPO (Optuna) + ensemble + MLflow
├── tests/
│   └── test_pipeline.py     ← Pytest unit tests
├── models/                  ← Output: submission.csv (git-ignored)
├── .github/workflows/
│   └── ci.yml               ← GitHub Actions CI
├── Makefile
├── requirements.txt
└── README.md
```


---

## 🧠 Models

### MLP
Deep network with BatchNorm + Dropout. Units halved every 2 layers for a funnel structure. Trained with AdamW + EarlyStopping + ReduceLROnPlateau.

### Wide & Deep
Combines a linear (wide) path for memorization with a deep path for generalization. Outputs are summed before the final sigmoid.

---

## 📊 Feature Engineering

| Feature | Description |
|---------|-------------|
| `Deck`, `Cabin_num`, `Side` | Parsed from `Cabin` field |
| `Group_size`, `Is_alone` | Derived from `PassengerId` group |
| `Total_spend` | Sum of all spending columns |
| `Log_total_spend` | Log-transformed total spend |
| `Spend_per_room` | RoomService / (TotalSpend + 1) |
| `Any_spend` | Binary: did passenger spend anything? |
| `Age_bin` | Age bucketed into 5 categories |
| `CryoSleep_spend` | Interaction: CryoSleep × TotalSpend |

---

## 🚀 Quickstart

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/spaceship-titanic.git
cd spaceship-titanic
```

### 2. Install
```bash
pip install -r requirements.txt
# or
make install
```

### 3. Get Data
Download from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and place CSVs in `data/`.

```bash
# With Kaggle CLI:
kaggle competitions download -c spaceship-titanic -p data/
unzip data/spaceship-titanic.zip -d data/
```

### 4. Run EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 5. Train
```bash
make train           # 30 Optuna trials, 5 folds
make train-fast      # 10 trials, 3 folds (quick test)
```

### 6. View MLflow Experiments
```bash
make mlflow-ui       # opens http://localhost:5000
```

### 7. Run Tests
```bash
make test
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| OOF AUC | ~0.88 |
| OOF Accuracy | ~80% |
| Kaggle LB | ~0.80+ |

---

## 🔬 Hyperparameters Tuned (Optuna)

| Parameter | Search Space |
|-----------|-------------|
| `n_layers` | 2 – 6 |
| `units` | 64, 128, 256, 512 |
| `dropout` | 0.1 – 0.5 |
| `lr` | 1e-4 – 1e-2 (log) |
| `batch_size` | 128, 256, 512 |
| `activation` | relu, swish, gelu |
| `l2_reg` | 1e-5 – 1e-2 (log) |

---

## 🛠 Tech Stack

- **TensorFlow / Keras** — MLP & Wide&Deep models
- **Optuna** — Bayesian hyperparameter optimization (TPE)
- **MLflow** — experiment tracking & artifact logging
- **scikit-learn** — preprocessing pipeline & cross-validation
- **Pandas / NumPy** — data manipulation
- **GitHub Actions** — CI (lint + test on Python 3.10 & 3.11)

---


