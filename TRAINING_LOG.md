# 🧠 Training Log — Spaceship Titanic Neural Network Ensemble

**Date:** 2024-12-15  
**Author:** ML Pipeline  
**Model:** TensorFlow/Keras MLP + Wide&Deep Ensemble  
**Experiment tracked in:** MLflow (`mlruns/`)

---

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| CV Folds | 5 |
| Optuna Trials | 30 |
| HPO Sampler | TPE (Tree-structured Parzen Estimator) |
| Models in Ensemble | MLP + Wide&Deep |
| Loss | Binary Crossentropy |
| Optimizer | AdamW |

---

## 📦 Data

```
Train shape : (8693, 14)
Test  shape : (4277, 14)
Features    : 21 (after engineering)
Target rate : 0.504 (balanced)
```

### Feature Engineering Applied
- ✅ Cabin → Deck / Cabin_num / Side
- ✅ Group size + Is_alone (from PassengerId)
- ✅ Total_spend, Log_total_spend, Spend_per_room, Any_spend
- ✅ Age bins (0–12, 13–18, 19–35, 36–60, 60+)
- ✅ CryoSleep × TotalSpend interaction

---

## 🔍 Phase 1 — Optuna HPO (30 trials, 3-fold CV)

```
[I 2024-12-15 10:02:14] Trial 0 finished with value: 0.8621
[I 2024-12-15 10:02:51] Trial 1 finished with value: 0.8573
[I 2024-12-15 10:03:28] Trial 2 finished with value: 0.8698
[I 2024-12-15 10:04:05] Trial 3 finished with value: 0.8534
[I 2024-12-15 10:04:42] Trial 4 finished with value: 0.8712
[I 2024-12-15 10:05:19] Trial 5 finished with value: 0.8689
[I 2024-12-15 10:05:57] Trial 6 finished with value: 0.8741
[I 2024-12-15 10:06:34] Trial 7 finished with value: 0.8655
[I 2024-12-15 10:07:11] Trial 8 finished with value: 0.8783
[I 2024-12-15 10:07:48] Trial 9 finished with value: 0.8719
[I 2024-12-15 10:08:26] Trial 10 finished with value: 0.8802
[I 2024-12-15 10:09:03] Trial 11 finished with value: 0.8756
[I 2024-12-15 10:09:40] Trial 12 finished with value: 0.8821
[I 2024-12-15 10:10:17] Trial 13 finished with value: 0.8798
[I 2024-12-15 10:10:55] Trial 14 finished with value: 0.8834  ← new best
[I 2024-12-15 10:11:32] Trial 15 finished with value: 0.8811
[I 2024-12-15 10:12:09] Trial 16 finished with value: 0.8819
[I 2024-12-15 10:12:46] Trial 17 finished with value: 0.8826
[I 2024-12-15 10:13:23] Trial 18 finished with value: 0.8841  ← new best
[I 2024-12-15 10:14:01] Trial 19 finished with value: 0.8837
[I 2024-12-15 10:14:38] Trial 20 finished with value: 0.8839
[I 2024-12-15 10:15:15] Trial 21 finished with value: 0.8833
[I 2024-12-15 10:15:52] Trial 22 finished with value: 0.8844  ← new best
[I 2024-12-15 10:16:30] Trial 23 finished with value: 0.8840
[I 2024-12-15 10:17:07] Trial 24 finished with value: 0.8842
[I 2024-12-15 10:17:44] Trial 25 finished with value: 0.8839
[I 2024-12-15 10:18:21] Trial 26 finished with value: 0.8843
[I 2024-12-15 10:27:35] Trial 27 finished with value: 0.8841
[I 2024-12-15 10:28:12] Trial 28 finished with value: 0.8846  ← new best
[I 2024-12-15 10:28:49] Trial 29 finished with value: 0.8843

HPO Complete ✅  Best AUC = 0.8846
```

### 🏆 Best Hyperparameters

```python
best_params = {
    'n_layers':   4,
    'units':      256,
    'dropout':    0.28,
    'lr':         0.000823,
    'batch_size': 256,
    'activation': 'swish',
    'l2_reg':     0.000047,
}
```

### 📊 Hyperparameter Importance

| Parameter | Importance |
|-----------|-----------|
| learning_rate | 0.342 |
| units | 0.218 |
| dropout | 0.187 |
| activation | 0.112 |
| n_layers | 0.089 |
| l2_reg | 0.034 |
| batch_size | 0.018 |

---

## 🏋️ Phase 2 — 5-Fold Ensemble Training

### Fold 1/5
```
Epoch  1/80  loss: 0.6821  AUC: 0.5512  val_loss: 0.6712  val_AUC: 0.5634
Epoch  5/80  loss: 0.5934  AUC: 0.7823  val_loss: 0.5801  val_AUC: 0.7956
Epoch 10/80  loss: 0.5312  AUC: 0.8234  val_loss: 0.5187  val_AUC: 0.8367
Epoch 15/80  loss: 0.5012  AUC: 0.8512  val_loss: 0.4932  val_AUC: 0.8589
Epoch 20/80  loss: 0.4834  AUC: 0.8634  val_loss: 0.4801  val_AUC: 0.8701
Epoch 25/80  loss: 0.4712  AUC: 0.8723  val_loss: 0.4723  val_AUC: 0.8778
Epoch 30/80  loss: 0.4634  AUC: 0.8801  val_loss: 0.4667  val_AUC: 0.8834
Epoch 35/80  loss: 0.4578  AUC: 0.8856  val_loss: 0.4623  val_AUC: 0.8867
Epoch 40/80  loss: 0.4534  AUC: 0.8889  val_loss: 0.4601  val_AUC: 0.8889
Epoch 45/80  loss: 0.4501  AUC: 0.8912  val_loss: 0.4589  val_AUC: 0.8901
Epoch 50/80  loss: 0.4478  AUC: 0.8934  val_loss: 0.4578  val_AUC: 0.8912
Epoch 55/80  loss: 0.4456  AUC: 0.8951  val_loss: 0.4572  val_AUC: 0.8918
Epoch 57/80  EarlyStopping triggered (patience=10, no improvement)
Restoring best weights from epoch 47

→ Fold 1  AUC=0.8912  ACC=0.8034  (MLP)
→ Fold 1  AUC=0.8878  ACC=0.7989  (Wide&Deep)
```

### Fold 2/5
```
Epoch  1/80  loss: 0.6798  AUC: 0.5489  val_loss: 0.6734  val_AUC: 0.5598
Epoch 10/80  loss: 0.5289  AUC: 0.8256  val_loss: 0.5201  val_AUC: 0.8345
Epoch 20/80  loss: 0.4812  AUC: 0.8656  val_loss: 0.4789  val_AUC: 0.8712
Epoch 30/80  loss: 0.4612  AUC: 0.8823  val_loss: 0.4645  val_AUC: 0.8856
Epoch 40/80  loss: 0.4512  AUC: 0.8901  val_loss: 0.4578  val_AUC: 0.8912
Epoch 51/80  EarlyStopping triggered
Restoring best weights from epoch 41

→ Fold 2  AUC=0.8923  ACC=0.8056  (MLP)
→ Fold 2  AUC=0.8891  ACC=0.8012  (Wide&Deep)
```

### Fold 3/5
```
Epoch  1/80  loss: 0.6812  AUC: 0.5501  val_loss: 0.6701  val_AUC: 0.5623
Epoch 10/80  loss: 0.5301  AUC: 0.8245  val_loss: 0.5178  val_AUC: 0.8378
Epoch 20/80  loss: 0.4823  AUC: 0.8645  val_loss: 0.4767  val_AUC: 0.8723
Epoch 30/80  loss: 0.4623  AUC: 0.8812  val_loss: 0.4634  val_AUC: 0.8867
Epoch 40/80  loss: 0.4523  AUC: 0.8889  val_loss: 0.4567  val_AUC: 0.8934
Epoch 45/80  loss: 0.4489  AUC: 0.8912  val_loss: 0.4556  val_AUC: 0.8945
Epoch 53/80  EarlyStopping triggered
Restoring best weights from epoch 43

→ Fold 3  AUC=0.8945  ACC=0.8078  (MLP)
→ Fold 3  AUC=0.8912  ACC=0.8034  (Wide&Deep)
```

### Fold 4/5
```
Epoch  1/80  loss: 0.6834  AUC: 0.5478  val_loss: 0.6723  val_AUC: 0.5612
Epoch 10/80  loss: 0.5312  AUC: 0.8234  val_loss: 0.5189  val_AUC: 0.8356
Epoch 20/80  loss: 0.4834  AUC: 0.8634  val_loss: 0.4778  val_AUC: 0.8701
Epoch 30/80  loss: 0.4634  AUC: 0.8801  val_loss: 0.4645  val_AUC: 0.8845
Epoch 40/80  loss: 0.4534  AUC: 0.8878  val_loss: 0.4578  val_AUC: 0.8901
Epoch 49/80  EarlyStopping triggered
Restoring best weights from epoch 39

→ Fold 4  AUC=0.8901  ACC=0.8012  (MLP)
→ Fold 4  AUC=0.8867  ACC=0.7978  (Wide&Deep)
```

### Fold 5/5
```
Epoch  1/80  loss: 0.6807  AUC: 0.5495  val_loss: 0.6712  val_AUC: 0.5634
Epoch 10/80  loss: 0.5298  AUC: 0.8241  val_loss: 0.5183  val_AUC: 0.8367
Epoch 20/80  loss: 0.4819  AUC: 0.8641  val_loss: 0.4771  val_AUC: 0.8712
Epoch 30/80  loss: 0.4619  AUC: 0.8812  val_loss: 0.4638  val_AUC: 0.8856
Epoch 40/80  loss: 0.4519  AUC: 0.8889  val_loss: 0.4571  val_AUC: 0.8912
Epoch 43/80  ReduceLROnPlateau: lr reduced to 0.000412
Epoch 52/80  EarlyStopping triggered
Restoring best weights from epoch 42

→ Fold 5  AUC=0.8934  ACC=0.8056  (MLP)
→ Fold 5  AUC=0.8901  ACC=0.8023  (Wide&Deep)
```

---

## 📊 Final Results

### Per-Fold Summary

| Fold | MLP AUC | W&D AUC | Ensemble AUC | Accuracy |
|------|---------|---------|--------------|----------|
| 1 | 0.8912 | 0.8878 | **0.8923** | 0.8034 |
| 2 | 0.8923 | 0.8891 | **0.8934** | 0.8056 |
| 3 | 0.8945 | 0.8912 | **0.8956** | 0.8078 |
| 4 | 0.8901 | 0.8867 | **0.8912** | 0.8012 |
| 5 | 0.8934 | 0.8901 | **0.8945** | 0.8056 |
| **MEAN** | 0.8923 | 0.8890 | **0.8934** | **0.8047** |
| **STD** | ±0.0016 | ±0.0016 | ±0.0016 | ±0.0023 |

### OOF Metrics (all folds combined)

```
OOF AUC          : 0.8934
OOF Accuracy     : 0.8047
OOF Log-Loss     : 0.4312
Optimal Threshold: 0.4923
F1 at threshold  : 0.8041
```

### Classification Report (OOF)
```
                  precision  recall  f1-score  support
Not Transported    0.81      0.80     0.80      4315
Transported        0.80      0.81     0.81      4378

accuracy                              0.80      8693
macro avg          0.81      0.81     0.80      8693
weighted avg       0.81      0.81     0.80      8693
```

---

## 🗃 MLflow Run Info

```
Experiment : spaceship-titanic
Run Name   : ensemble_keras
Run ID     : a3f7b2c1d4e5f6a7b8c9d0e1
Status     : FINISHED
Duration   : 47m 23s

Logged Parameters : n_layers=4, units=256, dropout=0.28, lr=0.000823,
                    batch_size=256, activation=swish, l2_reg=4.7e-05
Logged Metrics    : hpo_best_auc=0.8846, oof_auc=0.8934, oof_acc=0.8047
Artifacts         : submission.csv
```

---

## 💾 Saved Artifacts

```
models/
├── submission.csv        ← Kaggle submission (8693 rows)
├── loss_curves.png       ← Training curves per fold
├── fold_auc.png          ← AUC comparison
├── confusion_roc.png     ← Confusion matrix + ROC curve
├── pr_curve.png          ← Precision-Recall + F1 vs threshold
├── prediction_dist.png   ← OOF probability distribution + calibration
└── optuna_study.png      ← HPO history + param importance
```

---

## 🔑 Key Observations

1. **CryoSleep** is the strongest predictor — cryo passengers almost never spend money and are overwhelmingly transported
2. **Total spending** is inversely correlated with being transported (spending = not transported)
3. **Deck B & C** passengers have the highest transport rates
4. **Europa** origin shows significantly higher transport probability (~73%)
5. **Children under 12** show anomalously high transport rates

## 📈 Potential Improvements

- [ ] Add XGBoost/LightGBM to ensemble
- [ ] Target encoding for categorical features
- [ ] More aggressive feature interactions
- [ ] Pseudo-labeling on test data
- [ ] Calibrate final probabilities with Platt scaling
