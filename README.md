# 🚀 SPACESHIP TITANIC PREDICTION: MASTERING THE ML PIPELINE

## Project Summary

This project focuses on the Binary Classification problem of predicting whether a passenger was "Transported" or not following the anomaly on the Spaceship Titanic. The core objective was to build a robust model by meticulously following and optimizing the **7-Step Machine Learning Workflow**.

**🎯 Final Result:** An optimized accuracy of **79.24%** was achieved on the validation set using the **XGBoost** model.

---

## 🛠️ Methodology (The 7-Step Workflow)

### 1. Data Exploration and Preparation (EDA)
* **Initial Findings:** Identified severe data **skewness** in spending features and confirmed **CryoSleep** as a critical predictor.
* **Baseline Model:** Established a starting score using **Logistic Regression** (76.77% Accuracy).

### 2. Data Preprocessing & Feature Engineering
* **Handling Skewness:** Applied **`np.log1p`** to all spending columns to compress outliers and normalize the distribution.
* **Feature Engineering:** Extracted the critical **`Deck`** and **`Side`** features from the raw `Cabin` column, which significantly improved the model's predictive power.
* **Scaling:** Applied **`StandardScaler`** to all numerical features (`Age`, Log-Transformed Spending) to ensure fair comparison (mean=0, std=1).

### 3. Optimization and Model Selection
* **Iteration 1 (Random Forest):** Switched from a linear (Logistic Regression) to a non-linear (Random Forest) model to capture complex feature interactions.
* **Hyperparameter Tuning (Grid Search):** Used **Grid Search** to find the optimal balance for the Random Forest model. The resulting best parameter, **`max_depth=8`**, was key to preventing **overfitting**.
* **Feature Selection:** Used **Feature Importance** to identify and **drop low-value/noisy features** (e.g., `Deck_T`, `VIP_True`), leading to the final optimized score.
* **Final Model:** The final submission utilizes the **XGBoost Classifier** for its superior performance in ensemble learning.

---

## 📚 Technical Requirements
* Python 3.x
* pandas, numpy
* scikit-learn
* xgboost

---

## 📝 2. Essential Notebook Adjustments

You must add **Markdown cells** and **code comments** to your Jupyter Notebook to explain *why* you made certain decisions.

### **A. Documenting Log Transform (The "Why"):**

Explain the purpose of the `log1p` line.

```python
# [Markdown Cell: 2.1 Transformation for Skewness]

# PURPOSE: CURE SKEWNESS (Lệch)
# We use np.log1p(x) (which is log(x+1)) on spending columns because the original data
# is heavily skewed (many zeros, massive maximums). Log transformation
# compresses the large outliers, leading to a more normal distribution for the model.

skewed_columns = ['VRDeck','Spa','ShoppingMall','FoodCourt','RoomService']
df_train[skewed_columns] = df_train[skewed_columns].apply(np.log1p)
