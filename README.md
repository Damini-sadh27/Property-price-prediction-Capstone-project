# 🏡 Predicting Property Prices Using Machine Learning

> Capstone Project — an end-to-end regression pipeline that predicts residential property sale prices from structural, qualitative, and locational features, achieving a **test R² of ~0.93 (93%)** with a Gradient Boosting Regressor.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![status](https://img.shields.io/badge/status-complete-brightgreen)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Objectives](#-project-objectives)
- [Workflow](#-workflow)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Tech Stack](#-tech-stack)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)
- [Citations](#-citations)
- [License](#-license)

---

## 🔎 Overview

The real estate market is shaped by a wide mix of factors — location, size, quality, and amenities — which makes accurate price prediction genuinely useful to agents, buyers, sellers, and lenders. This project builds a complete machine learning pipeline that:

- Cleans and imputes a **1,460-row, 80-feature** residential property dataset
- Performs deep **exploratory data analysis (EDA)** to uncover price drivers
- Engineers **domain-informed features** (total living area, house age, total bathrooms, etc.)
- Correctly encodes **ordinal vs. nominal** categorical variables
- Applies **scaling and PCA** for dimensionality reduction
- Trains and compares **five regression algorithms**
- Selects and interprets the **best-performing model**

**Best model:** Gradient Boosting Regressor
**Test R²:** ≈ 0.93 | **MAE:** ≈ $13,893 | **RMSE:** ≈ $19,665

---

## 🗂 Dataset

| File | Description |
|---|---|
| `train.csv` / `Property_data.csv` | Training set — 1,460 property records, 80 features + target |
| `test.csv` | Test set (holdout, unlabeled) |
| `data_description.txt` | Full description of every column |
| `sample_submission.csv` | Benchmark submission (linear regression on a few basic features) |

**Target variable:** `PropPrice` — the property's sale price in dollars.

**Feature groups include:**
- **Location & zoning** — `Neighborhood`, `PropertyZone`, `Condition1/2`
- **Size & structure** — `PropertySize`, `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `BsmntSqFtage`
- **Quality & condition** — `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`
- **Age** — `YearBuilt`, `YearRemodAdd`
- **Amenities** — `PoolArea`, `CntFireplaces`, garage/basement details
- **Sale details** — `SaleMon`, `YrSold`, `SaleType`, `SaleCondn`

> A complete field-by-field description of all 80 columns is included in `data_description.txt` / the report.

---

## 🎯 Project Objectives

1. Collect and clean real estate data from a specific location.
2. Handle **ordinal** and **nominal** columns separately using the metadata to correctly identify each.
3. Use **scaling, PCA, and `fillna()`** techniques to handle missing data and improve accuracy.
4. Perform **EDA** to identify key variables influencing property prices.
5. Apply the correct **encoding technique** for ordinal vs. nominal variables.
6. Develop a machine learning model to predict property prices.
7. Evaluate model performance and compare across algorithms.
8. Present findings and insights clearly.

*Note: No feature columns were dropped other than the identifier column (`PropertyID`). Target R² range: 75–85% (achieved: 93%).*

---

## 🔁 Workflow

```
Data Collection
      │
Data Cleaning (drop ID, check nulls/duplicates)
      │
Missing Value Treatment (None-fill / median / mode / 0-fill)
      │
Feature Engineering (HouseAge, TotalSF, TotalBath, TotalPorchSF, ...)
      │
Exploratory Data Analysis (Skewness/Kurtosis, Univariate, Bivariate, Multivariate)
      │
Outlier Detection & Treatment (IQR method, GrLivArea outlier removal, capping)
      │
Encoding (Ordinal mapping + One-Hot Encoding)
      │
Feature Scaling & PCA (StandardScaler, 95% variance retained)
      │
Train-Test Split (80/20)
      │
Model Training (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
      │
Model Evaluation (R², MAE, RMSE)
      │
Best Model Selection (Gradient Boosting)
      │
Model Interpretation (Actual vs Predicted, Feature Importance)
```

---

## 📁 Repository Structure

```
property-price-prediction/
│
├── Prop_price_prediction.ipynb     # Main Jupyter Notebook (full pipeline)
├── Property_data.csv               # Training dataset
├── data_description.txt            # Column-level data dictionary
├── Capstone_Project_1_Report.docx  # Project Summary Report (Word)
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/property-price-prediction.git
cd property-price-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## ▶️ Usage

```bash
# Launch Jupyter Notebook
jupyter notebook Prop_price_prediction.ipynb
```

Run all cells top to bottom. The notebook is organized into clearly labeled sections (Data Loading, Cleaning, EDA, Feature Engineering, Encoding, Scaling/PCA, Modeling, Evaluation) so you can also run it section-by-section.

To use the trained model on new data:

```python
import pandas as pd
import numpy as np

# new_data must go through the same cleaning / feature engineering / encoding
# pipeline used in the notebook before being passed to the model
preds_log = best_model.predict(new_data_processed)
preds = np.expm1(preds_log)   # inverse of the log1p transform applied to the target
```

---

## 🧪 Methodology

### 1. Data Cleaning
- Dropped only the `PropertyID` identifier column.
- Checked for duplicates (`0` found) and inspected data types (`df.info()`).

### 2. Missing Value Treatment
| Strategy | Applied to |
|---|---|
| `"None"` fill | Categorical columns where NaN means *feature absent* (`PoolQC`, `Alley`, `BasementType`, ...) |
| Median fill | `PropertyFrontage` (numerical, skewed) |
| Mode fill | `Electrical` (single missing value) |
| `0` fill | `ExteriorCladdingArea`, `BasementYrBlt` (no cladding / no basement) |

### 3. Feature Engineering
```python
df["HouseAge"]      = df["SaleYr"] - df["YearBuilt"]
df["RemodAge"]       = df["SaleYr"] - df["YearRemodAdd"]
df["TotalSF"]        = df["BsmntSqFtage"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"]      = df["Bath1"] + 0.5*df["Bath2"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
df["TotalPorchSF"]   = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
df["HasPool"]        = (df["PoolArea"] > 0).astype(int)
df["HasGarage"]      = (df["BasementCars"] > 0).astype(int)
df["HasFireplace"]   = (df["CntFireplaces"] > 0).astype(int)
```

### 4. Outlier Treatment
- Removed 4 documented `GrLivArea > 4000` outliers (large area, unexpectedly low price).
- Applied **IQR-based capping** (not row removal) to continuous features such as `PropertySize`, `ExteriorCladdingArea`, `BsmntSqFtage`, `1stFlrSF`, `2ndFlrSF`, `GrLivArea`.

### 5. Encoding
- **Ordinal** quality columns (`ExterQual`, `KitchenQual`, `BasementQual`, ...) → mapped to ordered integers (`Po`=1 … `Ex`=5) to preserve ranking.
- **Nominal** columns (`Neighborhood`, `BldgType`, `SaleType`, ...) → one-hot encoded with `drop_first=True`.

### 6. Scaling & Dimensionality Reduction
- Target log-transformed (`np.log1p`) to correct right-skew.
- Features standardized with `StandardScaler`.
- **PCA** retained 95% of variance using **141 of 231** components.

### 7. Model Training
Five regression models were trained and evaluated with a consistent function that inverse-transforms predictions (`np.expm1`) before scoring:

| Model | Trained on |
|---|---|
| Linear Regression | Scaled features |
| Ridge Regression (α=10) | Scaled features |
| Lasso Regression (α=0.001) | Scaled features |
| Random Forest (300 trees, depth 15) | Raw features |
| Gradient Boosting (300 trees, lr=0.05, depth 3) | Raw features |

---

## 📊 Results

| Model | R² Score | MAE ($) | RMSE ($) |
|---|---|---|---|
| 🏆 **Gradient Boosting** | **0.9300** | **13,892.80** | **19,665.11** |
| Lasso Regression | 0.9188 | 14,675.95 | 21,177.45 |
| Ridge Regression | 0.9094 | 15,474.33 | 22,376.50 |
| Linear Regression | 0.9081 | 15,671.68 | 22,527.63 |
| Random Forest | 0.9005 | 16,388.13 | 23,439.34 |

The **Gradient Boosting Regressor** was selected as the final model — its actual-vs-predicted plot shows predictions clustering tightly along the diagonal, and its feature-importance ranking highlights `TotalSF`, `OverallQual`, `GrLivArea`, and `TotalBath` as the dominant predictors.

---

## 💡 Key Insights

- **Living area and quality dominate price.** `GrLivArea`, `TotalSF`, and `OverallQual` show the strongest positive correlation with `PropPrice`.
- **Location matters a lot.** Neighborhoods like `NoRidge`, `NridgHt`, `StoneBr`, and `Timber` command significantly higher median prices than `IDOTRR`, `MeadowV`, and `BrDale`.
- **Newer homes sell higher.** `HouseAge` is negatively correlated with price.
- **The target is right-skewed.** Most homes sell under ~$250,000, with a long tail of luxury properties — a `log1p` transform stabilizes this before modeling.
- **High-cardinality categorical space is compressible.** One-hot encoding expanded the feature space to 231 columns, but 95% of the variance is captured in just 141 PCA components.
- **Tree-based ensembles win.** Gradient Boosting outperforms linear/regularized models by capturing non-linear interactions (e.g., quality × size).

---

## 🛠 Tech Stack

- **Language:** Python 3.13
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
  - `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`
  - `StandardScaler`, `PCA`, `train_test_split`
  - `r2_score`, `mean_absolute_error`, `mean_squared_error`
- **Environment:** Jupyter Notebook

---

## 🎓 Learning Outcomes

- Cleaning real-world tabular data and distinguishing *missing* vs. *structurally absent* values.
- Applying context-appropriate imputation (median / mode / constant fills).
- Conducting structured EDA (skewness/kurtosis, univariate, bivariate, multivariate).
- Correctly encoding ordinal vs. nominal categorical variables.
- Detecting and treating outliers with the IQR method (removal vs. capping).
- Engineering domain-informed features that improve model performance.
- Using scaling and PCA for dimensionality reduction.
- Training, tuning, and comparing multiple regression algorithms under a consistent evaluation framework.
- Interpreting a fitted model via actual-vs-predicted plots and feature importance.

---

## 🚀 Future Improvements

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV` or Bayesian optimization.
- Try `XGBoost`, `LightGBM`, or `CatBoost` for potentially stronger gradient-boosted performance.
- Stacking/ensembling the top models for a further accuracy boost.
- Deploy the final model behind a simple API (e.g., Flask/FastAPI) for live predictions.
- Add SHAP-based explainability for individual predictions, not just global feature importance.

---

## 📚 Citations

- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3).
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning.* Springer.
- McKinney, W. *Python for Data Analysis.* O'Reilly Media.
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [seaborn documentation](https://seaborn.pydata.org/)
- [Kaggle: House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📄 License

This project is released under the [MIT License](LICENSE) — feel free to use, modify, and share with attribution.

---

*If you found this project useful, consider giving it a ⭐ on GitHub!*
