#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 07:50:21 2
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
"""df=pd.read_csv("C:\\Users\\ricky\\Downloads\\Dash\\datosDiscretizados.csv", sep = ",")
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])

df= df.drop('thalach', axis=1) """

np.random.seed(42)

# # Dataframe con datos de prueba
#df_20 = df.sample(frac=0.2)
#df_20.to_csv('datos_20.csv', index=False, sep=',')


# Dataframe con datos de entrenamiento
#df_80 = df.drop(df_20.index)
#df_80.to_csv('datos_80.csv', index=False, sep=',')

# df_20=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_20.csv", sep = ",")
# df_80=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 2/Proyecto 2/datos_80.csv", sep = ",")


# # Seleccionar características y variable objetivo
# X = df.drop("num", axis=1)
# y = df["num"]

# # Dividir datos en conjuntos de entrenamiento y prueba
# X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2)


#------------------------------------------------------------------------------
dff=pd.read_csv("C:\\Users\\ricky\\Downloads\\Dash\\datos_20.csv")
# Red Bayesiana Modelo entrenado
dff= dff.drop(['age', "sex", "trestbps", "chol", "fbs",
              "restecg"], axis=1) 
model = BayesianNetwork ([
                          ( "exang" , "cp" ),
                          ( "exang" , "num" ),
                          ( "num" , "thal" ),
                          ( "num" , "ca" ),
                          (  "num", "oldpeak" ),
                          ( "num", "slope"  ),
])



emv = MaximumLikelihoodEstimator( model = model , data = dff )
# Estimar para nodos sin padres


# Estimar para nodo exang
cpdem_exang = emv.estimate_cpd( node ="exang")
print( cpdem_exang )
model.fit(data=dff , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

#------------------------------------------------------------------------------

# Se realiza la inferencia sobre el modelo 
# Se colocan las predicciones en una lista


inference = VariableElimination(model)
#predict= []
predicciones = []

for i, row in dff.iterrows():
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

reales = dff['num'].tolist()

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


#------------------------------------------------------------------------------

# Reporte de clasificación
report = classification_report(reales, predicciones)
print("Reporte de clasificación:")
print(report)






