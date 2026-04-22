# Credit Card Customer Churn Prediction with Machine Learning

## Project Overview
This project focuses on building a predictive system to identify high-risk customers likely to churn (attrite) from credit card services. Early identification allows retail banks to implement proactive retention strategies, which is **5-25x more cost-effective** than acquiring new customers.

## Methodology
The project follows a comprehensive data science pipeline to ensure model robustness and business interpretability:

1. **Data Preprocessing:** Handled missing values, treated outliers, and performed feature scaling using `StandardScaler`.
2. **Feature Engineering:** Developed strategic metrics such as `Avg_Trans_Value` and `Customer_Segment` to capture deep behavioral patterns.
3. **Clustering (K-Means):** Performed unsupervised learning to group customers into behavioral segments (e.g., VIP, Dormant, High-Debt).
4. **Dimensionality Reduction (PCA):** Addressed multicollinearity by compressing 20+ variables into 4 core Principal Components (PCs) while retaining key information.
5. **Model Training & Optimization:** Compared and tuned **Random Forest** and **Lasso-regularized Logistic Regression** models using **Grid Search CV** and **5-fold Cross-Validation**.

## Performance & Key Results
- **Overall Accuracy:** 95.76%
- **ROC-AUC Score:** **0.99** (Indicating near-ideal classification power)
- **Recall (Risk Group):** **79.3%** (Successfully identified ~80% of actual churners)
- **Critical Business Insight:** Transaction count (`Total_Trans_Ct`) was identified as the most significant early-warning indicator for churn risk.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Techniques:** Ensemble Learning (Random Forest), PCA, K-Means Clustering, L1 (Lasso) Regularization.

## Repository Structure
- `Code/`: Contains the Jupyter Notebook with end-to-end analysis and modeling.
- `Data/`: Reference to the "Credit Card Customers" dataset from Kaggle.
- `Images/`: Visualizations including Confusion Matrix and ROC Curves.
