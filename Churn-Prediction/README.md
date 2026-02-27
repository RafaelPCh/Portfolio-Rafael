# Customer Churn Prediction

🇪🇸 [Leer en español](./README-es.md)

## 1. Introduction: The Business Problem

The objective of this project is to predict customer churn in a telecommunications company. Identifying at-risk customers allows for the implementation of proactive retention strategies, reducing revenue loss and improving customer loyalty.

## 2. Dataset

The dataset contains customer information from a telecommunications company.

*   **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
*   **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
*   **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
*   **Target Variable:** `Churn` (Yes/No).

## 3. Project Workflow

The project followed a standard data science workflow:

1.  **Data Preprocessing:**
    *   Handling null values and inconsistencies (e.g., converting `TotalCharges` to numeric).
    *   Missing value imputation using the median (`SimpleImputer`) and scaling of numerical variables (`StandardScaler`).
    *   Categorical variable encoding using `OneHotEncoder`.

2.  **Model Training & Tuning:**
    *   Trained models:
        *   **Logistic Regression** (Baseline)
        *   **Random Forest Classifier**
        *   **XGBoost Classifier**
    *   Hyperparameter optimization was performed using `GridSearchCV` with stratified cross-validation (`StratifiedKFold`).

3.  **Evaluation & Threshold Optimization:**
    *   We prioritized **F1-Score** and **Recall** for the positive class (Churn) to minimize false negatives (customers who leave without detection).
    *   The decision threshold was optimized by maximizing the F1-Score on the validation set, rather than using the default 0.5.

## 4. Model Performance Comparison

Summary of the models' performance on the test set using the optimized threshold.

| Model | ROC AUC | PR AUC | F1-Score (Churn=Yes) | Precision (Churn=Yes) | Recall (Churn=Yes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.830 | 0.619 | 0.614 | 0.525 | 0.739 |
| **Random Forest** | **0.834** | **0.620** | **0.615** | **0.498** | **0.804** |
| XGBoost | 0.832 | 0.620 | 0.618 | 0.530 | 0.742 |

**Conclusion:** The **Random Forest** model achieved the best overall performance, especially excelling in **Recall (80.4%)**, meaning it is capable of identifying the vast majority of at-risk customers. While this implies a slight increase in false positives, the benefit is significantly higher for the objective of this project.

## 5. Key Drivers of Churn

The models (especially Logistic Regression and Random Forest) highlighted the following factors as significant predictors:

*   **Internet Service (Fiber Optic):** Customers with fiber optic have a higher churn probability, which could indicate issues with the service or competitive pricing.
*   **Contract Type:** Long-term contracts (Two year) significantly reduce the risk of churn.
*   **Tenure:** The longer a customer has been with the company, the less likely they are to churn.

Based on these results, we recommend implementing loyalty strategies for at-risk customers, along with improvements to the internet service.

## 6. Installation & Usage

### Requirements
This project uses **Python 3.13.5** and the following main libraries:
*   `pandas`, `numpy` (Data processing)
*   `matplotlib` (Visualization)
*   `scikit-learn`, `xgboost` (Modeling)

### How to Run
Execute the Jupyter Notebook `churn.ipynb` to reproduce the training, evaluation, and model selection.

## 7. Next Steps
*   **Feature Engineering:** Explore interactions between variables (e.g., `tenure` * `MonthlyCharges`).
*   **Deep Learning:** Experiment with simple Neural Networks to compare performance.

## 8. Project Structure
```text
├── churn.ipynb       # Main Jupyter Notebook (Analysis and Modeling)
├── churn_model.joblib # Trained Random Forest model
├── churn_train.csv   # Training dataset
├── churn_test.csv    # Test dataset
├── README.md         # Project documentation (English)
└── README-es.md      # Project documentation (Spanish)
```


