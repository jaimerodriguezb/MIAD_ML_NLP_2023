#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/images/banner_1.png)

# # Construcción e implementación de modelo Random Forest
# En este notebook aprenderá a construir e implementar un modelo de Random Forest usando la librería especializada sklearn. Así mismo aprenderá a calibrar los parámetros del modelo y a obtener la importancia de las variables para la predicción. 

# ## Instrucciones Generales:
# 
# El modelo de Random Forest que construirá por medio de este notebook deberá predecir si el salario de un beisbolista es alto (>425) dadas las variables de desempeño en su carrera. Por esta razón, los datos a usar en el notebook son los de las Grandes Ligas de Béisbol de las temporadas 1986 y 1987, para más detalles: https://rdrr.io/cran/ISLR/man/Hitters.html. 
#    
# Para realizar la actividad, solo siga las indicaciones asociadas a cada celda del notebook. 

# ## Importar base de datos y librerías

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# Carga de datos de archivos .csv
url = 'https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/hitters.csv'
hitters = pd.read_csv(url)

# Eliminación filas con observaciones nulas
hitters.dropna(inplace=True)
hitters.head()


# ## Codificar variables categóricas

# In[3]:


# Codificación de las variables categoricas
hitters['League'] = pd.factorize(hitters.League)[0]
hitters['Division'] = pd.factorize(hitters.Division)[0]
hitters['NewLeague'] = pd.factorize(hitters.NewLeague)[0]
hitters.head()


# In[4]:


# Selección de variables predictoras
feature_cols = hitters.columns[hitters.columns.str.startswith('C') == False].drop('Salary')
feature_cols


# ## Definir las varibles del problema - Predictoras y varible de respuesta

# In[5]:


# Separación de variables predictoras (X) y variable de interes (y)
X = hitters[feature_cols]
y = (hitters.Salary > 425).astype(int)


# ## Implementación modelo usando *Sklearn*

# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Definición de modelo Random Forest para un problema de clasificación
clf = RandomForestClassifier()
clf


# In[7]:


#Impresión de desempeño del modelo usando la función cross_val_score  (más detalles en https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
pd.Series(cross_val_score(clf, X, y, cv=10)).describe()


# ## Calibración de parámetros

# ### Calibración de n_estimators
# 
# **n_estimators** es la cantidad de árboles a contruir dentro del bosque aleatorio.

# In[8]:


# Creación de lista de valores para iterar sobre diferentes valores de n_estimators
estimator_range = range(10, 310, 10)

# Definición de lista para almacenar la exactitud (accuracy) promedio para cada valor de n_estimators
accuracy_scores = []

# Uso de un 5-fold cross-validation para cada valor de n_estimators
for estimator in estimator_range:
    clf = RandomForestClassifier(n_estimators=estimator, random_state=1, n_jobs=-1)
    accuracy_scores.append(cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean())


# In[9]:


# Gráfica del desempeño del modelo vs la cantidad de n_estimators
plt.plot(estimator_range, accuracy_scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')


# ### Calibracióm max_features
# 
# **max_features** es el número de variables que se deben considerar en cada árbol.

# In[ ]:


# Creación de lista de valores para iterar sobre diferentes valores de max_features
feature_range = range(1, len(feature_cols)+1)

# Definición de lista para almacenar la exactitud (accuracy) promedio para cada valor de max_features
accuracy_scores = []

# Uso de un 10-fold cross-validation para cada valor de max_features
for feature in feature_range:
    clf = RandomForestClassifier(n_estimators=200, max_features=feature, random_state=1, n_jobs=-1)
    accuracy_scores.append(cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean())


# In[ ]:


# Gráfica del desempeño del modelo vs la cantidad de max_features
plt.plot(feature_range, accuracy_scores)
plt.xlabel('max_features')
plt.ylabel('Accuracy')


# ## Implementación de un Random Forest con los mejores parámetros

# In[ ]:


# Definición del modelo con los parámetros max_features=6 y n_estimators=200 
clf = RandomForestClassifier(n_estimators=200, max_features=6, random_state=1, n_jobs=-1)
clf.fit(X, y)


# In[ ]:


# Impresión de resultados de desemepeño del modelo
pd.DataFrame({'feature':feature_cols, 'importance':clf.feature_importances_}).sort_values('importance')


# In[ ]:




