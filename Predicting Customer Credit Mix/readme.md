# Credit Mix Prediction using Machine Learning

This project builds and evaluates multiple machine learning models to predict customer **Credit Mix** categories (Bad, Standard, Good) using banking and financial data. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and actionable business recommendations.

---

##  Project Objectives

* Clean and preprocess raw banking data
* Perform exploratory data analysis to understand trends and anomalies
* Train and compare multiple ML models
* Optimize models using hyperparameter tuning
* Evaluate final model performance using cross-validation
* Provide business recommendations based on model insights

---

##  Dataset

* **Source:** Public Bank Dataset (GitHub)
* **Target Variable:** `Credit_Mix` (Bad, Standard, Good)
* **Key Features:**

  * Age, Annual_Income, Monthly_Inhand_Salary
  * Num_Bank_Accounts, Num_Credit_Card
  * Interest_Rate, Outstanding_Debt
  * Num_of_Delayed_Payment, Credit_History_Age_Months
  * Credit_Utilization_Ratio, Total_EMI_per_month
  * Occupation, Payment_Behaviour

---

##  Workflow Overview

### 1. Data Preprocessing

* Replaced invalid values with NaN
* Converted numeric columns to proper types
* Parsed `Credit_History_Age` into months
* Handled missing values (median for numeric, mode for categorical)
* One-hot encoded categorical variables
* Scaled numeric features using `StandardScaler`
* Created dummy variables for loan types
* Clipped outliers using IQR and domain limits

Output: `preprocessed_bank_data.csv`

---

### 2. Exploratory Data Analysis (EDA)

* Descriptive statistics
* Correlation heatmap
* Distribution plots
* Boxplots by credit mix
* Scatter plots for key relationships

**Key Findings:**

* Higher debt and frequent delayed payments are strongly linked to Bad credit mix
* Longer credit history is associated with Good credit mix
* Significant correlations exist between debt, interest rate, and delayed payments

---

### 3. Model Building

Models trained:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

Preprocessing:

* Numeric scaling using `StandardScaler`
* One-hot encoding for categorical features

Train–test split: 80% / 20%

---

### 4. Model Evaluation

Metrics used:

* Accuracy
* Precision (Macro)
* Recall (Macro)
* F1-score (Macro)
* ROC-AUC (One-vs-Rest)

Visualizations:

* Confusion matrices
* ROC-AUC scores
* Model comparison bar plots
* Radar chart
* Feature importance (RF and XGBoost)

**Best Base Model:** Random Forest

---

### 5. Hyperparameter Tuning

Search Methods:

* GridSearchCV (Logistic Regression)
* RandomizedSearchCV (Random Forest, XGBoost)

Scoring Metric: `f1_macro`

Outputs:

* Best parameters for each model
* Cross-validated F1-scores
* Training time comparison

---

### 6. Retraining with Optimal Parameters

* Retrained tuned models on full training data
* Evaluated on unseen test set
* Generated classification reports and confusion matrices

---

### 7. Cross-Validation

* 5-fold Stratified Cross-Validation
* Metrics:

  * CV F1 Mean ± Std
  * CV Accuracy Mean ± Std

**Final Selected Model:** XGBoost (highest CV and test F1-score)

---

##  Key Results

* **Best Model:** XGBoost
* **F1-Score (Macro):** ~0.95
* **ROC-AUC:** ~0.99

**Top Important Features:**

* Outstanding_Debt
* Interest_Rate
* Num_of_Delayed_Payment
* Credit_History_Age_Months

---

## Business Insights

* Customers with higher debt and delayed payments are more likely to fall into the Bad credit mix
* Longer credit history significantly increases the likelihood of Good credit mix
* Credit utilization ratio and interest rate are strong risk indicators

---

##  Recommendations Based on Model Insights

* Prioritize key variables (Outstanding Debt, Interest Rate, Delayed Payments, Credit History Age) in risk assessment
* Implement payment reminder systems to reduce delays
* Offer debt management programs for high-risk customers
* Provide rewards or incentives for customers with good credit profiles

---

##  Actionable Steps to Improve Outcomes

### Data Improvement

* Collect real-time transaction and spending behavior data
* Apply SMOTE or ADASYN to handle class imbalance
* Anonymize sensitive fields such as SSN

### Model Enhancement

* Explore neural networks for complex patterns
* Use ensemble stacking (XGBoost + Random Forest)
* Apply early stopping and stronger cross-validation

### Ethical Considerations

* Check for bias across age and occupation
* Add fairness metrics
* Include human review for high-risk predictions

### Future Work

* Incorporate time-series analysis using monthly data
* Apply data augmentation
* Scale and refine the system over a 3–6 month period

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn

---

##  Project Structure

```
├── data/
│   └── preprocessed_bank_data.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_retraining.ipynb
│   └── 07_cross_validation.ipynb
├── outputs/
│   ├── confusion_matrices.png
│   ├── model_comparison_plots.png
│   ├── feature_importance.png
│   └── hyperparameter_tuning_results.png
├── README.md
└── requirements.txt
```

---

##  Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for predicting customer credit mix. The XGBoost model achieved the best overall performance and provided valuable insights into key financial risk factors. The system can be further enhanced with additional data, fairness evaluation, and real-time deployment strategies.

---

##  Contact

For questions or collaboration:

* **Name:** Sojib Chandra Roy
* **Role:** ML Enthusiast
* **Mail:** rcsojib.cse!@gmail.com

