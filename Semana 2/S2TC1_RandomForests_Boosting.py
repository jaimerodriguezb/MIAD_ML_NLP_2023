#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/images/banner_1.png)

# # Taller: Construcción e implementación de modelos Bagging, Random Forest y XGBoost
# 
# En este taller podrán poner en práctica sus conocimientos sobre la construcción e implementación de modelos de Bagging, Random Forest y XGBoost. El taller está constituido por 8 puntos, en los cuales deberan seguir las intrucciones de cada numeral para su desarrollo.

# ## Datos predicción precio de automóviles
# 
# En este taller se usará el conjunto de datos de Car Listings de Kaggle donde cada observación representa el precio de un automóvil teniendo en cuenta distintas variables como año, marca, modelo, entre otras. El objetivo es predecir el precio del automóvil. Para más detalles puede visitar el siguiente enlace: [datos](https://www.kaggle.com/jpayne/852k-used-car-listings).

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importación de librerías
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, accuracy_score, explained_variance_score
from xgboost import XGBRegressor, plot_importance


# Lectura de la información de archivo .csv
data = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTrain_carListings.zip')

data.head()


# In[3]:


# Preprocesamiento de datos para el taller
data = data.loc[data['Model'].str.contains('Camry')].drop(['Make', 'State'], axis=1)
data = data.join(pd.get_dummies(data['Model'], prefix='M'))
data = data.drop(['Model'], axis=1)

# Visualización dataset
data.head()


# In[4]:


# Separación de variables predictoras (X) y variable de interés (y)
y = data['Price']
X = data.drop(['Price'], axis=1)


# In[5]:


# Separación de datos en set de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Punto 1 - Árbol de decisión manual
# 
# En la celda 1 creen un árbol de decisión **manualmente**  que considere los set de entrenamiento y test definidos anteriormente y presenten el RMSE y MAE del modelo en el set de test.

# In[6]:


# Celda 1
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[7]:


def build_tree(X, y, max_depth, min_samples_split, depth=0):
    n_samples, n_features = X.shape
    
    # Comprobar si hemos alcanzado la profundidad máxima o si tenemos menos muestras que las muestras mínimas para dividir
    if depth >= max_depth or n_samples < min_samples_split:
        return np.mean(y)

    # Encontrar la característica y el umbral que minimizan el error
    best_loss = np.inf
    best_feature, best_threshold = None, None
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            y_left = y[X[:, feature] < threshold]
            y_right = y[X[:, feature] >= threshold]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            loss = np.sum((y_left - np.mean(y_left))**2) + np.sum((y_right - np.mean(y_right))**2)
            if loss < best_loss:
                best_loss = loss
                best_feature = feature
                best_threshold = threshold

    # Verificar si no pudimos encontrar una división que reduzca el error
    if best_feature is None:
        return np.mean(y)

    # Dividir los datos en nodos secundarios izquierdo y derecho
    left_indices = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[~left_indices], y[~left_indices]

    # Crear los nodos secundarios izquierdo y derecho
    left = build_tree(X_left, y_left, max_depth, min_samples_split, depth+1)
    right = build_tree(X_right, y_right, max_depth, min_samples_split, depth+1)

    # Crear el nodo actual
    return {'feature': best_feature, 'threshold': best_threshold, 'left': left, 'right': right}


# In[8]:


# Construcción del árbol
max_depth = 5
min_samples_split = 10
tree = build_tree(X.values, y.values, max_depth, min_samples_split)

print(tree)


# In[9]:


def predict_sample(x, tree):
    # Función auxiliar para predecir una muestra utilizando el árbol de decisión
    if x[tree['feature']] < tree['threshold']:
        if isinstance(tree['left'], dict):
            return predict_sample(x, tree['left'])
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict_sample(x, tree['right'])
        else:
            return tree['right']

def predict(X, tree):
    # Función para predecir un conjunto de datos utilizando el árbol de decisión
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_pred[i] = predict_sample(X[i], tree)
    return y_pred

# Predecir los valores de y utilizando el conjunto de prueba y el árbol de decisión
y_pred = predict(X_test, tree)


# In[10]:


# Calcular RMSE y MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


# In[11]:


reg_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
reg_tree.fit(X_train, y_train)


# In[12]:


y_pred = reg_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[13]:


# Calcular el RMSE y MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print('RMSE:', rmse)
print('MAE:', mae)


# In[14]:


# Ajustar una línea de tendencia
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)

# Graficar la dispersión y la línea de tendencia
plt.scatter(y_test, y_pred)
plt.plot(y_test, p(y_test), "r--")
plt.xlabel('Precios reales')
plt.ylabel('Precios predichos')
plt.title('RMSE')
plt.show()


# ### Punto 2 - Bagging manual
# 
# En la celda 2 creen un modelo bagging **manualmente** con 10 árboles de clasificación y comenten sobre el desempeño del modelo.

# In[15]:


# Se crea un arreglo de 1 a 20
np.random.seed(1)

# Impresión de arreglo y muestreo aleatorio
nums = np.arange(1, 21)
print('Arreglo:', nums)
print('Muestreo aleatorio: ', np.random.choice(a=nums, size=20, replace=True))


# In[16]:


# Creación de 10 muestras de bootstrap 
np.random.seed(123)

n_samples = X_train.shape[0]
n_B = 10

samples = [np.random.choice(a=n_samples, size=n_samples, replace=True) for _ in range(1, n_B +1 )]
samples


# In[17]:


# Construcción un árbol de decisión para cada muestra boostrap

# Definición del modelo usando DecisionTreeRegressor de sklearn
treereg = DecisionTreeRegressor(max_depth=None, random_state=123)


# In[18]:


# DataFrame para guardar las predicciones de cada árbol
y_pred = pd.DataFrame(columns=['Price'])

# Entrenamiento de un árbol sobre cada muestra boostrap y predicción sobre los datos de test
for i, sample in enumerate(samples):
    X_train = X_train.iloc[sample, 0:]
    y_train = y_train.iloc[sample]
    treereg.fit(X_train, y_train)
    y_pred_i = treereg.predict(X_test)
    y_pred['Price' + str(i+1)] = y_pred_i


# In[19]:


y_pred = y_pred.drop(['Price'], axis=1)


# In[20]:


# Desempeño de cada árbol
for i in range(n_B):
    print('Árbol ', i, 'tiene un error: ', np.sqrt(mean_squared_error(y_pred.iloc[:,i], y_test)))


# In[21]:


# Predicciones promedio para cada obserbación del set de test
y_pred.mean(axis=1)


# In[22]:


# Error al promediar las predicciones de todos los árboles
np.sqrt(mean_squared_error(y_test, y_pred.mean(axis=1)))


# ### Punto 3 - Bagging con librería
# 
# En la celda 3, con la librería sklearn, entrenen un modelo bagging con 10 árboles de clasificación y el parámetro `max_features` igual a `log(n_features)` y comenten sobre el desempeño del modelo.

# In[23]:


features = X.columns
n_features=len(features)

clfBag = BaggingRegressor(n_estimators=10, max_features=max(1, int(log(n_features))))
clfBag.fit(X_train, y_train)


# In[24]:


prediccionclfBag = clfBag.predict(X_test)
RMSE_clfBag = np.sqrt(mean_squared_error(y_test, prediccionclfBag))

print('\nRMSE Bagging:')
print(RMSE_clfBag)


# ### Punto 4 - Random forest con librería
# 
# En la celda 4, usando la librería sklearn entrenen un modelo de Randon Forest para clasificación  y comenten sobre el desempeño del modelo.

# In[25]:


clfRF = RandomForestRegressor(n_estimators=100, max_features=n_features, random_state=1, n_jobs=-1)
clfRF.fit(X_train, y_train)


# In[26]:


prediccionclfRF = clfRF.predict(X_test)
RMSE_clfRF = np.sqrt(mean_squared_error(y_test, prediccionclfRF))

print('\nRMSE Random Forest sin calibrar:')
print(RMSE_clfRF)


# ### Punto 5 - Calibración de parámetros Random forest
# 
# En la celda 5, calibren los parámetros max_depth, max_features y n_estimators del modelo de Randon Forest para clasificación, comenten sobre el desempeño del modelo y describan cómo cada parámetro afecta el desempeño del modelo.


# #### Calibración 
# In[ ]:

estimator_range = range(20, 200, 20)
feature_range = range(1, X_train.shape[1]+1)

scores = {}

for depth in range(1,11):
    scores[depth] = []
    for feature in feature_range:
        scores[depth].append({'estimator':[], 'metric':[]})
        for estimator in estimator_range:
            clf = RandomForestRegressor(max_depth=depth, n_estimators=estimator, max_features=feature, 
                                        random_state=42, n_jobs=-1)
            scores[depth][feature-1]['estimator'].append(estimator)
            scores[depth][feature-1]['metric'].append(abs(cross_val_score(clf, X_train, y_train, cv=10, 
                                            scoring='neg_mean_absolute_error').mean()))

# In[ ]:
fig, axs = plt.subplots(5, 2, figsize=(10, 20))

depth_count = 0
for i in range(5):
    for j in range(2):
        depth_count += 1
        for features, score in enumerate(scores[depth_count]):
            axs[i,j].plot(score['estimator'],
                        score['metric'],
                        label=str(features + 1))      
        axs[i,j].set(xlabel='n_estimators - Depth {} [#features]'.format(depth_count), ylabel='mae')
        axs[i,j].legend()

# #### Calibración n_estimators

# In[ ]:

# Creación de lista de valores para iterar sobre diferentes valores de n_estimators
estimator_range = range(20, 200, 20)

nmse_scores = []

# Uso de un 5-fold cross-validation para cada valor de n_estimators
for estimator in estimator_range:
    clf = RandomForestRegressor(n_estimators=estimator, 
                                random_state=0, n_jobs=-1)
    nmse_scores.append(abs(cross_val_score(clf, X, y, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))

# Gráfica del desempeño del modelo vs la cantidad de n_estimators
plt.plot(estimator_range, nmse_scores)
plt.xlabel('n_estimators')
plt.ylabel('abs NMSE')

# El NMSE tuvo su min en n_estimators=100


# #### Calibración max_features

# In[29]:


feature_range = range(1, X_train.shape[1]+1)
nmse_scores = []

# Uso de un 10-fold cross-validation para cada valor de max_features
for feature in feature_range:
    clf = RandomForestRegressor(max_features=feature, n_estimators=100, random_state=0, n_jobs=-1)
    nmse_scores.append(abs(cross_val_score(clf, X_train, y_train, cv=10, 
                                           scoring='neg_mean_absolute_error')).mean())

# Gráfica del desempeño del modelo vs la cantidad de max_features
plt.plot(feature_range, nmse_scores)
plt.xlabel('max_features')
plt.ylabel('abs NMSE')

# El abs NMSE tuvo su min en max_features=8




# #### Calibración max_depth

# In[27]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

depth_range = range(1,31)

# Definición de lista para almacenar el score para cada valor de max_depth
nmse_scores = []

# Uso de un 5-fold cross-validation para cada valor de max_depth
for depth in depth_range:
    #clf = RandomForestRegressor(max_depth=depth, random_state=42, n_jobs=-1)
    clf = RandomForestRegressor(max_depth=depth, n_estimators=120, max_features=8, random_state=42, n_jobs=-1)
    nmse_scores.append(abs(cross_val_score(clf, X_train, y_train, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))


# Gráfica del desempeño del modelo vs la cantidad de max_depth
plt.plot(depth_range, nmse_scores)
plt.xlabel('max_depth')
plt.ylabel('abs NMSE')

# El abs NMSE tuvo su min en max_depth=20



# #### Implementación del Random Forest con los mejores parámetros


# In[ ]:
# Chequeo del Desempeño de conjunto de entrenamiento vs validación
clf = RandomForestRegressor(max_depth=20, max_features=8, n_estimators=100, random_state=42, n_jobs=-1)
print('MEA conjunto entrenaimento: ',abs(cross_val_score(clf, X_train, y_train, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))
print('MEA conjunto validación: ',abs(cross_val_score(clf, X_test, y_test, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))

# La gran diferencia muestra que hay overfitting


# In[ ]:
# Tuning the hyper paremeters
# Aplicando heuristica para max_features y probando max_depth menores, se encuentra que con:
# 
#  max_depth=5
#  max_features=3
# 
# ## El MEA de entrenaimento vs validción se acerca bastante


clf = RandomForestRegressor(max_depth=5, max_features=3, n_estimators=100, random_state=42, n_jobs=-1)
print('MEA conjunto entrenaimento: ',abs(cross_val_score(clf, X_train, y_train, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))
print('MEA conjunto validación: ',abs(cross_val_score(clf, X_test, y_test, cv=10, 
                                           scoring='neg_mean_absolute_error').mean()))

# In[ ]:

# Definición del modelo con los parámetros calibrados
clf = RandomForestRegressor(max_depth=6, max_features=4, n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_test, y_test)

# Desempeño del modelo usando la función cross_val_score
pd.Series(cross_val_score(clf, X_test, y_test, cv=10)).describe()


# In[ ]:
# Impresión de importancia de las features
pd.DataFrame({'feature':X.columns, 'importance':clf.feature_importances_}).sort_values('importance')


# ### Punto 6 - XGBoost con librería
# 
# En la celda 6 implementen un modelo XGBoost de clasificación con la librería sklearn y comenten sobre el desempeño del modelo.

# In[30]:


# Implementando un Modelo XGBoost Clasificador con parametros por default
xgb_reg = XGBRegressor()
xgb_reg


# In[31]:


# Entrenamiento (fit) y desempeño del modelo XGBClassifier
xgb_reg.fit(X_train, y_train)

y_pred_xgb_default = xgb_reg.predict(X_test)

# Variance_score
print("Varianza: %f" % (explained_variance_score(y_pred_xgb_default, y_test)))

# RMSE
print("RMSE: %f" % (np.sqrt(mean_squared_error(y_test, y_pred_xgb_default))))

# MAE
print('MAE: %f' % (mean_absolute_error(y_test, y_pred_xgb_default)))


# In[32]:


plot_importance(xgb_reg)


# ### Punto 7 - Calibración de parámetros XGBoost
# 
# En la celda 7 calibren los parámetros learning rate, gamma y colsample_bytree del modelo XGBoost para clasificación, comenten sobre el desempeño del modelo y describan cómo cada parámetro afecta el desempeño del modelo.

# In[33]:


# Para realizar la calibración se utiliza una implementacion 'Grid Search Cross-Validation'
from sklearn.model_selection import GridSearchCV

xgb_reg = XGBRegressor()

# Definir la cuadrícula de parámetros
param_grid = {
    'gamma': [0, 0.1, 0.5],
    'learning_rate': [0.01, 0.1, 0.5],
    'colsample_bytree': [0.5, 0.7, 1],
    'n_estimators': [100],
}

# Se aplica la estrategia GridSearch
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros encontrados
print("Mejores parametros encontrados: ", grid_search.best_params_)
print("RMSE más bajo encontrado: ", np.sqrt(np.abs(grid_search.best_score_)))


# In[34]:


# Implementando un Modelo XGBoost Clasificador con parametros learning rate, gamma y colsample_bytree calibrados
xgb_reg_calibrado = XGBRegressor(gamma=0, learning_rate=0.1, colsample_bytree=0.5, n_estimators=100)
xgb_reg_calibrado


# In[35]:


# Entrenamiento (fit) y desempeño del modelo XGBClassifier Calibrado
xgb_reg_calibrado.fit(X_train, y_train)

y_pred_xgb_calibrado = xgb_reg_calibrado.predict(X_test)

# Variance_score
print("Varianza: %f" % (explained_variance_score(y_pred_xgb_calibrado, y_test)))

# RMSE
print("RMSE: %f" % (np.sqrt(mean_squared_error(y_test, y_pred_xgb_calibrado))))

# MAE
print('MAE: %f' % (mean_absolute_error(y_test, y_pred_xgb_calibrado)))


# In[ ]:


plot_importance(xgb_reg_calibrado)


# ### Punto 8 - Comparación y análisis de resultados
# En la celda 8 comparen los resultados obtenidos de los diferentes modelos (random forest y XGBoost) y comenten las ventajas del mejor modelo y las desventajas del modelo con el menor desempeño.

# In[ ]:





# In[ ]:


# Celda 8

