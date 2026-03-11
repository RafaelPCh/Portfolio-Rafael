# Predicción de Desarrollo Personal y Resiliencia en Personas con Discapacidad (ENEDIS)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow.svg)

## 📌 Resumen Ejecutivo
Este proyecto desarrolla un **Pipeline de Machine Learning** diseñado para predecir si la discapacidad que posee lo ayudo a lograr un desarrollo personal pleno, utilizando datos de la Encuesta Nacional Especializada sobre Discapacidad (ENEDIS).

Desde una óptica de **Políticas Públicas y Economía de la Salud**, el objetivo del modelo es identificar algorítmicamente a los individuos con mayor riesgo de *no lograr un desarrollo personal pleno*. Al focalizar los verdaderos predictores del bienestar integral, el hacedor de politica publica puede optimizar la asignación de recursos públicos, migrando de un modelo puramente asistencialista-económico hacia intervenciones que prioricen la salud mental y la inserción psicosocial efectiva.

## 📂 Origen de los Datos
El conjunto de datos analizado corresponde a la **Encuesta Nacional Especializada sobre Discapacidad (ENEDIS)**. 

* **Nota de Reproducibilidad:** Debido al peso de los archivos originales de la base de datos nacional, estos no se encuentran en este repositorio. Para replicar el análisis localmente, descargue los módulos correspondientes (.dbf o .csv) desde el portal oficial de datos públicos y colóquelos en la carpeta `/Data/` de la raíz del proyecto. url: https://proyectos.inei.gob.pe/microdatos/ 

## ⚙️ Arquitectura Técnica y Metodología
El proyecto sigue rigurosos estándares de la industria para garantizar la escalabilidad y prevenir la fuga de información (*Data Leakage*):

1. **Ingeniería de Características:** Limpieza de respuestas nulas y estandarización categórica.
2. **Transformadores Personalizados (Custom Transformers):** Creación de clases en Scikit-Learn (`BaseEstimator`, `TransformerMixin`) para inyectar reglas logicas demográficas (ej. restricciones para edad laboral) antes del modelado.
3. **Pipelines de Scikit-Learn:** Ensamblaje automatizado de preprocesamiento estadístico (`SimpleImputer`, `OneHotEncoder`) acoplado directamente al algoritmo predictivo.
4. **Optimización (Tuning):** Evaluación de Modelos Base (Baseline) y posterior ajuste fino del modelo ganador usando Validación Cruzada en Grilla (`GridSearchCV`).

## 📊 Análisis Económico y de Políticas Públicas: El Costo del Error
El modelo predictivo ganador (**Random Forest Optimizado**) alcanzó un ROC-AUC de 0.88. Sin embargo, en el diseño de políticas públicas, tenemos dos aristas sumamnete importantes. Por un lado, la eficente asignacion de los recursosdisponibles. Por otro lado, el impacto social de las decisiones tomadas. En ese sentido, es importante analizar los siguientes escenarios:

* **Falso Positivo (Predecir Resiliencia [Sí] cuando es [No]): Alto Costo Social y Económico de Largo Plazo.** 
  El modelo asume erróneamente que el individuo ha logrado forjar su desarrollo personal y no requiere intervención inmediata.
  * *Impacto:* El Estado niega/retrasa el soporte psicológico a una persona que necesitaria la asistencia para lograr su desarrollo personal.
  
* **Falso Negativo (Predecir Falta de Resiliencia [No] cuando es [Sí]): Costo de Ineficiencia Asignativa.** 
  El modelo asume que el individuo necesita intervención cuando, en realidad, ya posee las herramientas actitudinales necesarias para su desarrollo.
  * *Impacto:* Gasto ineficiente de fondos públicos (horas de profesionales, subsidios específicos) que podrían haberse destinado a casos de mayor criticidad.

**Logro del Modelo:** A través de la optimización de hiperparámetros (Tuning), logramos **reducir significativamente los Falsos Positivos** (disminución del error en la detección de la clase negativa). Esto significa que el modelo es altamente conservador y seguro, disminuyendo la probabilidad geométrica de abandonar a individuos que genuinamente necesitan apoyo psicológico.

## 🧠 Interpretabilidad (Explainable AI) y Conclusiones Centrales

Para dotar al modelo de explicabilidad, implementamos técnicas de interpretabilidad post-entrenamiento (XAI):

1. **La Psicología sobre la Biología:** El análisis de *Feature Importance* post-Pipeline determinó que la variable psicológica vinculada a la valoración actitudinal/personal tiene una injerencia predictiva casi **10 veces mayor** que cualquier otra variable, incluyendo el tipo de discapacidad física padecida. Funciona como el factor de peso primario en la arquitectura del árbol de decisión.
2. **La Curva de la Edad (PDP):** El *Partial Dependence Plot* reveló que la probabilidad de percepción de desarrollo personal dibuja una curva en forma de U invertida: es mínima durante la infancia, alcanza su pico máximo de resiliencia en la adultez productiva media (30 a 50 años), y sufre un marcado declive durante la tercera edad.

### 💡 Recomendación Estratégica
Este modelo sugiere un pivot estratégico para las políticas públicas institucionales: **las políticas de estado no deben enfocarse exclusivamente en resolver limitantes arquitectónicas o soluciones puramente monetarias**. Resulta imperativo priorizar la asistencia psicológica tanto en las edades tempranas como en las maduras para lograr la integración e independencia real y sostenible en personas con discapacidad.

## 🚀 Cómo ejecutar este proyecto
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repo.git
   ```
2. Instalar los requerimientos (se recomienda crear un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el Jupyter Notebook principal: `TP_F_v6 copy.ipynb`
