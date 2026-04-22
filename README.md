# Credit Card Customer Churn Prediction (ML & Financial Insights)

## Project Overview
This project develops a high-precision machine learning system to predict credit card churn risk using a dataset of **10,127 customers** and **23 behavioral features**. By identifying early warning signals, banks can proactively retain customers, targeting a potential profit growth of **25% to 95%** through improved retention.

## Methodology & Detailed Workflow
The project follows a rigorous data science pipeline, ensuring data integrity and model interpretability:

1. **Data Preprocessing:** 
    - Cleaned **10k+ records** (zero missing values/duplicates).
    - Applied **Z-score Standardization** (`StandardScaler`) and **One-hot Encoding** to transform categorical variables for algebraic computation.
2. **Feature Engineering:** 
    - Created **2 strategic features**: `Avg_Trans_Value` (Average transaction size) and `Customer_Segment` (Behavioral labels).
3. **Behavioral Segmentation (K-Means):** 
    - Segmented customers into **3 distinct clusters** (e.g., "Dormant" group showed a **22.48%** churn rate—3x higher than others).
4. **Dimensionality Reduction (PCA):** 
    - Reduced **20+ features** down to **4 Principal Components (PCs)**, explaining **~65%** of total variance. This successfully addressed perfect multicollinearity (corr = 1.0) between `Credit_Limit` and `Avg_Open_To_Buy`.
5. **Model Training & Hyperparameter Tuning:** 
    - Optimized **Random Forest** via **Grid Search CV** and **5-fold Cross-Validation**.
    - Best Params: `n_estimators: 200`, `max_depth: 20`, `max_features: 'sqrt'`.

## Performance & Key Results
- **Overall Accuracy:** **95.76%**
- **ROC-AUC Score:** **0.99** (Near-perfect classification performance)
- **Recall (Churn Class):** **79.3%** (Effectively capturing 8/10 at-risk customers).
- **Precision:** **93.25%** (Minimizing "False Alarms" for the marketing department).

## Business Insights & Actionable Signals
- **Critical Threshold:** Identified a "drop-off point" at **25-30 transactions/year**. Customers below this threshold have a churn probability exceeding **50%**.
- **The "High-Spend Churn" Paradox:** Discovered that churned customers often have a high `Avg_Trans_Value` (**$63.59**) just before closing their accounts, indicating a few large final transactions.
- **Retention Strategy:** Banks should prioritize budget for **Cluster 2 (Dormant)** using frequency-based incentives (cashback for daily utilities) rather than high-value luxury rewards.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- **Models:** Random Forest, Lasso-regularized Logistic Regression, K-Means, PCA.
