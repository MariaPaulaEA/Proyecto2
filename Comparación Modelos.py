#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:19:31 2023

@author: valeriarrondon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 07:50:21 2023

@author: valeriarrondon
"""
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

#------------------------------------------------------------------------------

# Se carga el dataframe discretizado
df=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 1/Proyecto/datosDiscretizados.csv", sep = ",")
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])

#df= df.drop('thalach', axis=1) 

np.random.seed(42)

# # Dataframe con datos de prueba
df_20 = df.sample(frac=0.2)
#df_20.to_csv('datos_20.csv', index=False, sep=',')


# Dataframe con datos de entrenamiento
df_80 = df.drop(df_20.index)
#df_80.to_csv('datos_80.csv', index=False, sep=',')

# df_20=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_20.csv", sep = ",")
# df_80=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_80.csv", sep = ",")


# # Seleccionar características y variable objetivo
# X = df.drop("num", axis=1)
# y = df["num"]

# # Dividir datos en conjuntos de entrenamiento y prueba
# X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2)


#------------------------------------------------------------------------------

# Red Bayesiana Modelo Proyecto 1

model = BayesianNetwork ([( "age" , "fbs" ),
                          ( "age" , "chol" ),
                          ( "age", "trestbps" ),
                          ( "age" , "thalach" ),
                          ( "sex", "chol" ),
                          ( "cp" , "thal" ),
                          ( "cp" , "slope" ),
                          ( "cp" , "num" ),
                          ( "cp", "restecg" ),
                          ( "trestbps" , "cp" ),
                          ( "chol" , "trestbps" ),
                          ( "fbs" , "trestbps" ),
                          ( "restecg", "slope" ),
                          ( "restecg" , "num" ),
                          ( "thalach" , "slope" ),
                          ( "exang" , "cp" ),
                          ( "exang" , "num" ),
                          ( "oldpeak" , "num" ),
                          ( "slope" , "oldpeak" ),
                          ( "num" , "thal" ),
                          ( "num" , "ca" )])


emv = MaximumLikelihoodEstimator( model = model , data = df_20 )
# Estimar para nodos sin padres

# Estimar para nodo age
cpdem_age = emv.estimate_cpd( node ="age")
print( cpdem_age )
# Estimar para nodo sex
cpdem_sex = emv.estimate_cpd( node ="sex")
print( cpdem_sex )
# Estimar para nodo exang
cpdem_exang = emv.estimate_cpd( node ="exang")
print( cpdem_exang )
model.fit(data=df_20 , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

#------------------------------------------------------------------------------

# Se realiza la inferencia sobre el modelo 
# Se colocan las predicciones en una lista


inference = VariableElimination(model)
#predict= []
predicciones = []

for i, row in df_20.iterrows():
    query = inference.query(variables=['num'], evidence={
        'age': row['age'],
        'sex': row['sex'],
        'cp': row['cp'],
        'trestbps': row['trestbps'],
        'chol': row['chol'],
        'fbs': row['fbs'],
        'restecg': row['restecg'],
        'exang': row['exang'],
        'oldpeak': row['oldpeak'],
        'slope': row['slope'],
        'thal': row['thal'],
        'ca': row['ca'],
        'thalach': row['thalach']
    })
    
    
    # Num
    max_query = None
    max_prob = 0
    
    for i, prob in enumerate(query.values):
        if prob > max_prob:
            max_query = i
            max_prob = prob
    predicciones.append(max_query)
    
    
    # Probabilidad
    
    # prediccion = 0
    # for valor in query.values:
    #     if valor > prediccion:
    #         prediccion = valor
    # predict.append(prediccion)
    
   
    # prediccion = 0
    # print("probabbilidad 0")
    # print(query.values[0])
    # print("probabbilidad 1")
    # print(query.values[1])
    # print("probabbilidad 2")
    # print(query.values[2])
    # print("probabbilidad 3")
    # print(query.values[3])
    # print("probabbilidad 4")
    # print(query.values[4])
    # print(query.values)
    # predicciones.append(prediccion)

# Se obtienen las predicciones
predicciones

#------------------------------------------------------------------------------

# Datos reales del dataframe de prueba

reales = df_20['num'].tolist()

#------------------------------------------------------------------------------

# Matriz de confusión
matriz = metrics.confusion_matrix(reales, predicciones)
print("Matriz de confusión:")
print(matriz)

#------------------------------------------------------------------------------

# Se calculan los %

vp = matriz[0, 0] / sum(matriz[0, :])
fp = matriz[1, 0] / sum(matriz[1, :])
vn = matriz[1, 1] / sum(matriz[1, :])
fn = matriz[0, 1] / sum(matriz[0, :])

print(f"Verdaderos positivos: {vp:.2f}")
print(f"Falsos negativos: {fn:.2f}")
print(f"Verdaderos negativos: {vn:.2f}")
print(f"Falsos positivos: {fp:.2f}")

# Crear DataFrame
data = {'Valor': ['Verdaderos positivos', 'Falsos negativos', 'Verdaderos negativos', 'Falsos positivos'],
        'Porcentaje': [vp, fn, vn, fp]}
df2 = pd.DataFrame(data)

# Mostrar tabla
print(df2)


#------------------------------------------------------------------------------

# Reporte de clasificación
report = classification_report(reales, predicciones)
print("Reporte de clasificación:")
print(report)



import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(reales, predicciones)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


import seaborn as sns
# Plot the confusion matrix.
sns.heatmap(confusion_matrix,
            annot=True)
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de Confusión',fontsize=15)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Crear matriz de confusión
cm = confusion_matrix(reales, predicciones)

# Configurar el mapa de colores
cmap = sns.color_palette("PiYG")

# Graficar matriz de confusión
sns.heatmap(cm, annot=True, cmap=cmap)

# Agregar etiquetas y título
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de confusión',fontsize=17)

# Mostrar gráfico
plt.show()











#------------------------------------------------------------------------------

# Se carga el dataframe discretizado
df=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/Discrete.csv", sep = ",")
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])

#df= df.drop('thalach', axis=1) 

np.random.seed(42)

# # Dataframe con datos de prueba
df_20 = df.sample(frac=0.2)
#df_20.to_csv('datos_20.csv', index=False, sep=',')


# Dataframe con datos de entrenamiento
df_80 = df.drop(df_20.index)


# Red Bayesiana Modelo del Otro Grupo

model = BayesianNetwork ([( "age" , "ca" ),
                          ( "age" , "trestbps" ),
                          ( "age", "num" ),
                          ( "age", "thalach" ),
                          ( "sex" , "thal" ),
                          ( "slope" , "num" ),
                          ( "slope", "oldpeak"),
                          ( "slope" , "thalach" ),
                          ( "thal" , "exang" ),
                          ( "thal" , "num" ),
                          ( "thal" , "oldpeak" ),
                          ( "ca" , "num" ),
                          ( "exang", "cp"),
                          ( "exang", "thalach" ),
                          ( "exang", "oldpeak" ),
                          ( "num", "cp"  ),
                          ( "num", "oldpeak" ),
                          
])


emv = MaximumLikelihoodEstimator( model = model , data = df_20 )
# Estimar para nodos sin padres

# Estimar para nodo age
cpdem_age = emv.estimate_cpd( node ="age")
print( cpdem_age )
# Estimar para nodo sex
cpdem_sex = emv.estimate_cpd( node ="sex")
print( cpdem_sex )
# Estimar para nodo exang
cpdem_exang = emv.estimate_cpd( node ="slope")
print( cpdem_exang )

model.fit(data=df_20 , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

#------------------------------------------------------------------------------

# Se realiza la inferencia sobre el modelo 
# Se colocan las predicciones en una lista


inference = VariableElimination(model)
#predict= []
predicciones = []

for i, row in df_20.iterrows():
    query = inference.query(variables=['num'], evidence={
        'age': row['age'],
        'sex': row['sex'],
        'cp': row['cp'],
        'trestbps': row['trestbps'],
        'exang': row['exang'],
        'oldpeak': row['oldpeak'],
        'slope': row['slope'],
        'thal': row['thal'],
        'ca': row['ca'],
        'thalach': row['thalach']
    })
    
    
    # Num
    max_query = None
    max_prob = 0
    
    for i, prob in enumerate(query.values):
        if prob > max_prob:
            max_query = i
            max_prob = prob
    predicciones.append(max_query)
    
    
    # Probabilidad
    
    # prediccion = 0
    # for valor in query.values:
    #     if valor > prediccion:
    #         prediccion = valor
    # predict.append(prediccion)
    
   
    # prediccion = 0
    # print("probabbilidad 0")
    # print(query.values[0])
    # print("probabbilidad 1")
    # print(query.values[1])
    # print("probabbilidad 2")
    # print(query.values[2])
    # print("probabbilidad 3")
    # print(query.values[3])
    # print("probabbilidad 4")
    # print(query.values[4])
    # print(query.values)
    # predicciones.append(prediccion)

# Se obtienen las predicciones
predicciones

#------------------------------------------------------------------------------

# Datos reales del dataframe de prueba

reales = df_20['num'].tolist()

#------------------------------------------------------------------------------

# Matriz de confusión
matriz = confusion_matrix(reales, predicciones)
print("Matriz de confusión:")
print(matriz)

#------------------------------------------------------------------------------

# Se calculan los %

vp = matriz[0, 0] / sum(matriz[0, :])
fp = matriz[1, 0] / sum(matriz[1, :])
vn = matriz[1, 1] / sum(matriz[1, :])
fn = matriz[0, 1] / sum(matriz[0, :])

print(f"Verdaderos positivos: {vp:.2f}")
print(f"Falsos negativos: {fn:.2f}")
print(f"Verdaderos negativos: {vn:.2f}")
print(f"Falsos positivos: {fp:.2f}")


# Crear DataFrame
data = {'Valor': ['Verdaderos positivos', 'Falsos negativos', 'Verdaderos negativos', 'Falsos positivos'],
        'Porcentaje': [vp, fn, vn, fp]}
df2 = pd.DataFrame(data)

# Mostrar tabla
print(df2)

#------------------------------------------------------------------------------

# Reporte de clasificación
report = classification_report(reales, predicciones)
print("Reporte de clasificación:")
print(report)



import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(reales, predicciones)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


import seaborn as sns
# Plot the confusion matrix.
sns.heatmap(confusion_matrix,
            annot=True)
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de Confusión',fontsize=15)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Crear matriz de confusión
cm = confusion_matrix(reales, predicciones)

# Configurar el mapa de colores
cmap = sns.color_palette("PiYG")

# Graficar matriz de confusión
sns.heatmap(cm, annot=True, cmap=cmap)

# Agregar etiquetas y título
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de confusión',fontsize=17)

# Mostrar gráfico
plt.show()



#------------------------------------------------------------------------------
# Modelo 2 entrenado

# Se carga el dataframe discretizado
df=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 1/Proyecto/datosDiscretizados.csv", sep = ",")
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])

#df= df.drop('thalach', axis=1) 

np.random.seed(42)

# # Dataframe con datos de prueba
df_20 = df.sample(frac=0.2)
#df_20.to_csv('datos_20.csv', index=False, sep=',')


# Dataframe con datos de entrenamiento
df_80 = df.drop(df_20.index)
#df_80.to_csv('datos_80.csv', index=False, sep=',')

# df_20=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_20.csv", sep = ",")
# df_80=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_80.csv", sep = ",")


# # Seleccionar características y variable objetivo
# X = df.drop("num", axis=1)
# y = df["num"]

# # Dividir datos en conjuntos de entrenamiento y prueba
# X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2)


#------------------------------------------------------------------------------

# Red Bayesiana Modelo 2

model = BayesianNetwork ([
                          ( "exang" , "cp" ),
                          ( "exang" , "num" ),
                          ( "num" , "thal" ),
                          ( "num" , "ca" ),
                          (  "num", "oldpeak" ),
                          ( "num", "slope"  ),
])


emv = MaximumLikelihoodEstimator( model = model , data = df_20 )
# Estimar para nodos sin padres


# Estimar para nodo exang
cpdem_exang = emv.estimate_cpd( node ="exang")
print( cpdem_exang )
model.fit(data=df_20 , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

#------------------------------------------------------------------------------

# Se realiza la inferencia sobre el modelo 
# Se colocan las predicciones en una lista


inference = VariableElimination(model)
#predict= []
predicciones = []

for i, row in df_20.iterrows():
    query = inference.query(variables=['num'], evidence={
        'cp': row['cp'],       
        'exang': row['exang'],
        'oldpeak': row['oldpeak'],
        'slope': row['slope'],
        'thal': row['thal'],
        'ca': row['ca']
    })
    
    
    # Num
    max_query = None
    max_prob = 0
    
    for i, prob in enumerate(query.values):
        if prob > max_prob:
            max_query = i
            max_prob = prob
    predicciones.append(max_query)
    
    
    # Probabilidad
    
    # prediccion = 0
    # for valor in query.values:
    #     if valor > prediccion:
    #         prediccion = valor
    # predict.append(prediccion)
    
   
    # prediccion = 0
    # print("probabbilidad 0")
    # print(query.values[0])
    # print("probabbilidad 1")
    # print(query.values[1])
    # print("probabbilidad 2")
    # print(query.values[2])
    # print("probabbilidad 3")
    # print(query.values[3])
    # print("probabbilidad 4")
    # print(query.values[4])
    # print(query.values)
    # predicciones.append(prediccion)

# Se obtienen las predicciones
predicciones

#------------------------------------------------------------------------------

# Datos reales del dataframe de prueba

reales = df_20['num'].tolist()

#------------------------------------------------------------------------------

# Matriz de confusión
matriz = confusion_matrix(reales, predicciones)
print("Matriz de confusión:")
print(matriz)

#------------------------------------------------------------------------------

# Se calculan los %

vp = matriz[0, 0] / sum(matriz[0, :])
fp = matriz[1, 0] / sum(matriz[1, :])
vn = matriz[1, 1] / sum(matriz[1, :])
fn = matriz[0, 1] / sum(matriz[0, :])

print(f"Verdaderos positivos: {vp:.2f}")
print(f"Falsos negativos: {fn:.2f}")
print(f"Verdaderos negativos: {vn:.2f}")
print(f"Falsos positivos: {fp:.2f}")


# Crear DataFrame
data = {'Valor': ['Verdaderos positivos', 'Falsos negativos', 'Verdaderos negativos', 'Falsos positivos'],
        'Porcentaje': [vp, fn, vn, fp]}
df2 = pd.DataFrame(data)

# Mostrar tabla
print(df2)

from sklearn import metrics
Accuracy = metrics.accuracy_score(reales, predicciones)
Precision = metrics.precision_score(reales, predicciones, average ='macro')
Sensitivity_recall = metrics.recall_score(reales, predicciones, average ='macro')
F1_score = metrics.f1_score(reales, predicciones, average ='macro')

print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"F1_score":F1_score})

report = classification_report(reales,predicciones)


#------------------------------------------------------------------------------

# Reporte de clasificación
report = classification_report(reales, predicciones)
print(report)



import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(reales, predicciones)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


import seaborn as sns
# Plot the confusion matrix.
sns.heatmap(confusion_matrix,
            annot=True)
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de Confusión',fontsize=15)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Crear matriz de confusión
cm = confusion_matrix(reales, predicciones)

# Configurar el mapa de colores
cmap = sns.color_palette("PiYG")

# Graficar matriz de confusión
sns.heatmap(cm, annot=True, cmap=cmap)

# Agregar etiquetas y título
plt.ylabel('Predicción',fontsize=13)
plt.xlabel('Real',fontsize=13)
plt.title('Matriz de confusión',fontsize=17)

# Mostrar gráfico
plt.show()




