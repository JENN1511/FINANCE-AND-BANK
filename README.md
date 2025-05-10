# Análisis de Riesgo Crediticio con Machine Learning

Este proyecto implementa un modelo de regresión logística para predecir el riesgo crediticio utilizando el conjunto de datos *German Credit Data*. Se incluye limpieza de datos, codificación, creación de una variable objetivo (`default`) y evaluación del modelo.

## 📁 Estructura del Proyecto

- `german_credit_data.csv`: Dataset original.
- `main.ipynb` o `main.py`: Código principal del análisis y modelado.
- `README.md`: Descripción general del proyecto.

## 🧪 Tecnologías Utilizadas

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (opcional para visualizaciones)

## 📊 Objetivo del Proyecto

Predecir si un cliente representa un riesgo crediticio (`default`) en base a variables como:

- Edad
- Sexo
- Ingresos estimados (Saving/Checking account)
- Monto de crédito
- Duración del crédito
- Propósito del crédito
- Tipo de vivienda
- Ocupación

## 🧼 Limpieza de Datos

Se reemplazaron valores vacíos y sospechosos (como `" "`, `"none"`, `"na"`, `"-"`) por `NaN`, y se imputaron con la moda en columnas categóricas.

## 🎯 Variable Objetivo

Se creó una variable `default` de manera aleatoria para fines prácticos, utilizando una distribución con 70% de clientes sin riesgo (`0`) y 30% con riesgo (`1`):  
```python
# Crear variable aleatoria artificial (solo práctica)
np.random.seed(42)
df['default'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

## Fuente de los Datos

El conjunto de datos utilizado en este proyecto proviene de Kaggle:  
[German Credit Data](https://www.kaggle.com/datasets/uciml/german-credit/data).

Este dataset contiene información sobre el historial crediticio de clientes alemanes, utilizada para evaluar su riesgo crediticio.
