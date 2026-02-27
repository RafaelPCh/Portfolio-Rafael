# Customer Churn Prediction

🇺🇸 [Read this in English](./README.md)

## 1. Introduction: The Business Problem

El objetivo de este proyecto es predecir la tasa de abandono de clientes (Churn) en una empresa de telecomunicaciones. Identificar a los clientes en riesgo permite implementar estrategias de retención proactivas, reduciendo la pérdida de ingresos y mejorando la fidelidad.

## 2. Dataset

El dataset contiene información de clientes de una compañía de telecomunicaciones.

*   **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
*   **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
*   **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
*   **Target Variable:** `Churn` (Yes/No).

## 3. Project Workflow

El proyecto siguió un flujo de trabajo estándar de ciencia de datos:

1.  **Data Preprocessing:**
    *   Manejo de valores nulos e inconsistencias (ej. conversión de `TotalCharges` a numérico).
    *   Imputación de valores faltantes usando la mediana (`SimpleImputer`) y escalado de variables numéricas (`StandardScaler`).
    *   Codificación de variables categóricas utilizando `OneHotEncoder`.

2.  **Model Training & Tuning:**
    *   Modelos entrenados:
        *   **Logistic Regression** (Baseline)
        *   **Random Forest Classifier**
        *   **XGBoost Classifier**
    *   Se utilizó `GridSearchCV` con validación cruzada estratificada (`StratifiedKFold`) para la optimización de hiperparámetros.

3.  **Evaluation & Threshold Optimization:**
    *   Se priorizó el **F1-Score** y el **Recall** para la clase positiva (Churn) para minimizar los falsos negativos (clientes que se van y no detectamos).
    *   Se optimizó el umbral de decisión (threshold) maximizando el F1-Score en el set de validación, en lugar de usar el defecto 0.5.

## 4. Model Performance Comparison

Resumen del rendimiento de los modelos en el set de prueba utilizando el umbral optimizado.

| Model | ROC AUC | PR AUC | F1-Score (Churn=Yes) | Precision (Churn=Yes) | Recall (Churn=Yes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.830 | 0.619 | 0.614 | 0.525 | 0.739 |
| **Random Forest** | **0.834** | **0.620** | **0.615** | **0.498** | **0.804** |
| XGBoost | 0.832 | 0.620 | 0.618 | 0.530 | 0.742 |

**Conclusion:** El modelo **Random Forest** obtuvo el mejor rendimiento general, destacando especialmente en **Recall (80.4%)**, lo que significa que es capaz de identificar a la gran mayoría de los clientes en riesgo de abandono. Si bien esto conlleva un aumento en la cantidad de falsos positivos, para el objetivo de este proyecto el beneficio es mayor.

## 5. Key Drivers of Churn

Los modelos (especialmente Regresión Logística y Random Forest) destacaron los siguientes factores como predictores significativos:

*   **Internet Service (Fiber Optic):** Los clientes con fibra óptica tienen una mayor probabilidad de abandono, lo que podría indicar problemas con el servicio o precios competitivos.
*   **Contract Type:** Los contratos a largo plazo (Two year) reducen significativamente el riesgo de churn.
*   **Tenure:** A mayor antigüedad del cliente, menor es la probabilidad de que abandone la compañía.


Con base en estos resultados, se recomienda implementar estrategias de fidelización para los clientes en riesgo de abandono, junto con mejoras en el servicio de internet.

## 6. Installation & Usage

### Requirements
Este proyecto utiliza **Python 3.13.5** y las siguientes librerías principales:
*   `pandas`, `numpy` (Procesamiento de datos)
*   `matplotlib` (Visualización)
*   `scikit-learn`, `xgboost` (Modelado)

### How to Run
Ejecutar el Jupyter Notebook `churn.ipynb` para reproducir el entrenamiento, la evaluación y la selección del modelo.

## 7. Next Steps
*   **Feature Engineering:** Explorar interacciones entre variables (ej. `tenure` * `MonthlyCharges`).
*   **Deep Learning:** Experimentar con Redes Neuronales simples para comparar rendimiento.

## 8. Project Structure
```text
├── churn.ipynb       # Jupyter Notebook principal (Análisis y Modelado)
├── churn_model.joblib # Modelo Random Forest entrenado
├── churn_train.csv   # Dataset de entrenamiento
├── churn_test.csv    # Dataset de prueba
├── README.md         # Documentación del proyecto (Inglés)
└── README-es.md      # Documentación del proyecto (Español)
```
