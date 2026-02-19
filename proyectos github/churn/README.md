# Customer Churn Prediction for a Telecom Company

## 1. Introduction: The Business Problem

Customer churn, the rate at which customers stop doing business with a company, is a critical metric for subscription-based services like telecom providers. It is far more expensive to acquire a new customer than to retain an existing one.

This project aims to build a machine learning model that accurately predicts which customers are most likely to churn. By identifying these at-risk customers, the business can proactively engage them with targeted retention strategies, such as special offers or improved customer service, ultimately reducing revenue loss.

## 2. Dataset

The dataset contains customer information from a fictional telecom company. It includes:
*   **Customer Demographics:** Gender, senior citizen status, partner, and dependent status.
*   **Services Subscribed:** Phone service, multiple lines, internet service, online security, online backup, etc.
*   **Account Information:** Tenure (how long they've been a customer), contract type, payment method, paperless billing, monthly charges, and total charges.
*   **Target Variable:** `Churn` (Yes/No), indicating whether the customer left within the last month.

## 3. Project Workflow

The project followed a standard data science workflow:

1.  **Data Preprocessing:**
    *   Handled data inconsistencies (e.g., converting `TotalCharges` to a numeric type).
    *   Built a robust `scikit-learn` pipeline to systematically impute missing values (using the median for numerical columns) and scale features.
    *   Applied One-Hot Encoding to categorical variables to prepare them for modeling.

2.  **Model Training & Tuning:**
    *   Three different classification models were trained and compared:
        *   **Logistic Regression** (as a baseline)
        *   **Random Forest Classifier**
        *   **XGBoost Classifier**
    *   `GridSearchCV` was used to perform extensive hyperparameter tuning for each model, ensuring optimal performance was achieved. The models were tuned based on the **ROC AUC** metric.

3.  **Evaluation & Threshold Optimization:**
    *   While accuracy is important, the primary business goal is to correctly identify as many potential churners as possible (maximizing **Recall** for the "Yes" class) without overwhelming the retention team with false positives (maintaining reasonable **Precision**).
    *   Therefore, the **F1-Score** was chosen as the key evaluation metric. The default prediction threshold of 0.5 was found to be suboptimal, resulting in many missed churners (high false negatives).
    *   For each model, an optimal prediction threshold was calculated by maximizing the F1-score on a cross-validated portion of the training data. This custom threshold provides a better balance between Precision and Recall for this specific business problem.

## 4. Model Performance Comparison

The final models were evaluated on the held-out test set. The performance metrics below are reported using the **F1-optimized threshold**.

| Model | ROC AUC | PR AUC | F1-Score (Churn=Yes) | Precision (Churn=Yes) | Recall (Churn=Yes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.830 | 0.619 | 0.614 | 0.525 | 0.739 |
| **Random Forest** | **0.834** | **0.620** | **0.615** | **0.498** | **0.804** |
| XGBoost | 0.832 | 0.620 | 0.618 | 0.530 | 0.742 |

**Conclusion:** All three models perform similarly well, with the **Random Forest** model achieving the highest Recall (80.4%), meaning it successfully identified the largest percentage of actual churners. This makes it a strong candidate for deployment, as the primary goal is to catch at-risk customers.

## 5. Key Drivers of Churn

The models consistently highlighted the following factors as the most significant predictors of customer churn:

*   **Contract Type:** Customers with **Month-to-Month contracts** are far more likely to churn than those with One-Year or Two-Year contracts. This is the single most important predictor.
*   **Tenure:** New customers (with low **tenure**) are at a much higher risk of churning.
*   **Internet Service:** Customers with **Fiber Optic** internet service have a higher churn rate, which may indicate issues with pricing, reliability, or perceived value for that specific service.

These insights suggest that retention efforts should focus on encouraging new customers to move to longer-term contracts and investigating the customer experience for fiber optic users.

## 6. How to Run This Project

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2.  Create a `requirements.txt` file and install the necessary dependencies:
    *(You should create this file by running `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries used are: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `xgboost`.

3.  Open and run the `churn.ipynb` Jupyter Notebook to see the full analysis.
