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
B. Documenting Standardization (The "Fairness")
Mục đích: Giải thích tại sao StandardScaler là cần thiết để đảm bảo sự công bằng giữa các đặc trưng số.

Python

# [Markdown Cell: 2.2 Standardization for Fairness]

# PURPOSE: ENSURE FAIRNESS (mean=0, std=1)
# Standardizing Age and the log-transformed spending columns is mandatory for linear models
# to ensure features with large scales (like the spending features) aren't seen as more
# important than features with smaller scales (like Age).

from sklearn.preprocessing import StandardScaler
numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
scaler = StandardScaler() 
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
C. Documenting Grid Search Result (The "Overfitting Fix")
Mục đích: Giải thích lý do tại sao các thông số tối ưu nhất lại tốt hơn các thông số mặc định (đây là bằng chứng của quá trình tối ưu hóa).

Python

# [Markdown Cell: 3.2 Optimization Insight: Max_Depth]

# KEY INSIGHT: The best parameters found by Grid Search led to the final accuracy of 78.67%
# The most critical parameter was:
# 'max_depth': 8 

# By restricting the tree's depth, we successfully prevented the model from learning
# the training noise (overfitting), enabling it to generalize better to the unseen validation data.
