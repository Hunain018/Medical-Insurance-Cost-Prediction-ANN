# Medical Insurance Cost Prediction — ANN Regression
 
This project trains an Artificial Neural Network (ANN) with TensorFlow/Keras
to predict individual medical insurance **charges** from the classic
`insurance.csv` dataset (1,338 records: age, sex, BMI, children, smoker,
region, charges).
 
The full, executed analysis is in **`insurance_analysis.ipynb`**. This README
summarizes the data exploration, modeling approach, and results.
 
---
 
## 1. Dataset Overview
 
- **Shape:** 1,338 rows × 7 columns
- **Columns:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`
- **Missing values:** none
- **Duplicate rows:** 1 (a 19-year-old male, BMI 30.59, 0 children, non-smoker,
  northwest region — removed before modeling, leaving 1,337 rows)
- **Target variable:** `charges` (continuous, USD), ranging from ~$1,122 to
  ~$63,770, mean ≈ $13,270, heavily right-skewed (a small number of very
  expensive cases pull the average well above the median of ~$9,382).
## 2. Key Observations from EDA
 
- **Smoking status is the dominant driver of cost.** Smokers pay roughly
  **$32,050** on average vs. **$8,434** for non-smokers — almost a 4x
  difference. `smoker` has the highest correlation with `charges` (≈0.79).
- **Age has a clear positive relationship with charges** (correlation ≈0.30):
  costs trend upward with age, visible as several roughly parallel diagonal
  bands in the age-vs-charges scatter plot (the bands correspond to
  smokers vs. non-smokers and BMI tiers).
- **BMI correlates moderately with charges** (≈0.20). The effect of BMI is
  much stronger for smokers than non-smokers — high-BMI smokers form the
  most expensive group in the dataset.
- **Sex and region have only weak relationships with charges** (correlations
  ≈0.06 and below). Males have a slightly higher mean charge than females,
  and the southeast region is marginally more expensive on average, but
  these differences are small compared to the smoking effect.
- **Number of children** has almost no linear relationship with charges
  (correlation ≈0.07).
## 3. Preprocessing Steps
 
1. Dropped the 1 duplicate row.
2. Encoded `sex` (`female`→0, `male`→1) and `smoker` (`no`→0, `yes`→1) as
   binary integers.
3. One-hot encoded `region` (4 categories → 3 dummy columns, dropping the
   first to avoid redundancy).
4. Split into train/test sets (80% / 20%, `random_state=42`) →
   **1,069 training rows / 268 test rows**, 8 input features.
5. Standardized all features with `StandardScaler`, fit only on the training
   set and applied to both train and test (to avoid data leakage).
## 4. Model Architecture
 
A small feed-forward ANN for regression:
 
| Layer        | Units | Activation |
|--------------|-------|------------|
| Input        | 8     | —          |
| Dense        | 32    | ReLU       |
| Dense        | 16    | ReLU       |
| Dense        | 8     | ReLU       |
| Output       | 1     | Linear     |
 
**Total parameters:** 961
 
## 5. Training Configuration
 
- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Metric:** Mean Absolute Error (MAE)
- **Validation split:** 20% of the training data
- **Batch size:** 32
- **Epochs:** up to 300, with `EarlyStopping` (patience=20 on validation
  loss, restoring the best weights)
- Training and validation loss curves track each other closely throughout
  training (see `plots/05_training_history.png`), indicating **no
  significant overfitting**. Early stopping did not trigger before 300
  epochs — loss was still decreasing very slowly at the end, so a longer
  training run could squeeze out marginal further gains.
## 6. Results on the Test Set
 
| Metric | ANN Model | Baseline (predict mean charge) |
|--------|-----------|---------------------------------|
| MAE    | **$3,134.06** | $9,861.80 |
| RMSE   | **$4,882.62** | $13,612.43 |
| R²     | **0.870**     | 0.000 (by definition) |
 
- The model explains about **87% of the variance** in insurance charges on
  unseen data, more than tripling the explanatory power of a naive baseline
  and roughly halving both MAE and RMSE compared to the baseline.
- The **predicted-vs-actual scatter plot** (`plots/06_predicted_vs_actual.png`)
  shows tight clustering along the diagonal for lower-cost cases (mostly
  non-smokers), but more scatter for the highest-cost cases. A subset of the
  most expensive policyholders (likely high-BMI smokers) are systematically
  under- or over-predicted, suggesting the model could benefit from an
  explicit `bmi * smoker` interaction feature.
- The **residual distribution** (`plots/07_residuals.png`) is roughly
  centered around zero but has heavier tails on the positive side,
  consistent with the right-skew of the target variable — the model
  occasionally under-predicts very high charges.
## 7. Comparison to a Smaller Baseline Network
 
A smaller network (16 → 8 → 1 units, 150 epochs, no early stopping) was
tried first and achieved MAE ≈ $4,616.89, RMSE ≈ $6,755.70, R² ≈ 0.75. The
deeper 32 → 16 → 8 → 1 architecture with early stopping (this notebook)
improved R² from 0.75 to **0.87** and cut MAE/RMSE by roughly 30-35%,
showing that additional capacity and more training time meaningfully help on
this dataset.
 
## 8. Saved Artifacts (`outputs/`)
 
| File | Description |
|------|-------------|
| `insurance_ann_model.keras` | Trained Keras model (architecture + weights) |
| `scaler.joblib` | Fitted `StandardScaler` used for preprocessing |
| `test_predictions.csv` | Test set features, actual charges, predicted charges, and residuals |
| `metrics.json` | Final evaluation metrics (MAE, RMSE, R², baseline metrics, epochs trained) |
 
Plots from the EDA and evaluation steps are saved in `plots/`.
 
## 9. Possible Next Steps
 
- Add an interaction feature between `bmi` and `smoker` (or `age` and
  `smoker`) to help the model separate the "expensive smoker" cluster from
  the rest.
- Apply a log transform to `charges` before training, since the target is
  right-skewed — this often improves regression performance and residual
  behavior.
- Try regularization (dropout, L2) or hyperparameter tuning (learning rate,
  layer sizes) to see if test performance improves further.
- Compare against tree-based models (e.g., Random Forest, Gradient Boosting),
  which often perform very well on this dataset and can provide feature
  importance for interpretability.
