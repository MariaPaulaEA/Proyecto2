import plotly.express as px
import pandas as pd
import itertools
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output

from pgmpy.sampling import BayesianModelSampling 
import pandas as pd
df=pd.read_csv("C:\\Users\\ricky\\Downloads\\Dash\\processed.cleveland.data.csv") 
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])
df= df.drop('thalach', axis=1) 
print(df.head()) 
print(df.describe()) 
print(df.columns)
# EDAD
df["age"] = pd.cut(df["age"], bins=4, labels=['29-39', '40-49', '50-59', '60-79'])


# PRESIÓN SANGUÍNEA EN REPOSO
intervalos2 = [0, 80, 120, 129, 139, 179, 600]
categorias2 = ['hipotensión', 'normal', 'elevada', 'hiptertensión nivel 1', 'hiptertensión nivel 2', 'crisis hipertensión']
df['trestbps'] = pd.cut(df['trestbps'], bins=intervalos2, labels=categorias2)

# OLDPEAK
intervalos3 = [0.0, 1.4, 2.5, 7]
categorias3 = ['baja', 'normal', 'terrible']
df['oldpeak'] = pd.cut(df['oldpeak'], bins=intervalos3, labels=categorias3)

# COLESTEROL
intervalos = [0, 200, 239, 600]
categorias = ['saludable', 'riesgoso', 'peligroso']
df['chol'] = pd.cut(df['chol'], bins=intervalos, labels=categorias)

df= df.dropna() 
print(df)

df.to_csv('datosDiscretizados.csv', index=False)

exclusiones = ['age', 'sex', 'angina', 'exang']
tuplas = []

for col in df.columns:
    if col not in exclusiones:
        tupla = (col, df['sex'])
        tuplas.append(tupla)

print(tuplas)

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
#whitelist = [("age", "chol"),("age", "fbs"), ( "age", "trestbps"),  ("sex", "chol"), ("chol","trestbps"), ("fbs", "trestbps"),("exang" , "cp")]
whitelist = [("age", "chol"),("age", "fbs"), ( "age", "trestbps"),  ("sex", "chol"),("exang" , "cp")]
#blacklist = [("thal","num"), ("oldpeak","num"), ("slope","num"), ("ca","num"), ("restecg","num")] 
#blacklist2 = [("thal","num") , ("oldpeak","num"), ("slope","num"),("ca","num"), ("restecg","num"),( "cp", "exang"),("slope","oldpeak"), ("ca", "age"), ("thal","sex"), ("num", "exang")]

blacklist3 = [('ca', 'age'),("thal","num") ,("oldpeak","num"), ("slope","num"),("ca","num"), ("restecg","num"),( "cp", "exang"),("slope","oldpeak"), ("thal","sex"), ("num", "exang"), ('num', 'sex'),('oldpeak', 'slope'),('exang', 'sex'),('sex','exang'), ("exang", "num"), ("chol","age"),("ca", "sex")]

exclusiones = ['exang']
tuplas = []
print(df['sex'])
for col in df.columns:
    if col not in exclusiones:
        tupla = (col, 'exang')
        tuplas.append(tupla)

#print(tuplas)
#Datos de entrenamiento
dff=pd.read_csv("C:\\Users\\ricky\\Downloads\\Dash\\datos_80.csv")

#blacklistDif= [('sex', 'age'), ('cp', 'age'), ('trestbps', 'age'), ('chol', 'age'), ('fbs', 'age'), ('restecg', 'age'), ('exang', 'age'), ('oldpeak', 'age'), ('slope', 'age'), ('ca', 'age'), ('thal', 'age'), ('num', 'age'), ('age', 'sex'), ('cp', 'sex'), ('trestbps', 'sex'), ('chol', 'sex'), ('fbs', 'sex'), ('restecg', 'sex'), ('exang', 'sex'), ('oldpeak', 'sex'), ('slope', 'sex'), ('ca', 'sex'), ('thal', 'sex'), ('num', 'sex'),('age', 'exang'), ('sex', 'exang'), ('cp', 'exang'), ('trestbps', 'exang'), ('chol', 'exang'), ('fbs', 'exang'), ('restecg', 'exang'), ('oldpeak', 'exang'), ('slope', 'exang'), ('ca', 'exang'), ('thal', 'exang'), ('num', 'exang')]
scoring_method = K2Score(data=dff) 
esth = HillClimbSearch(data=dff) 

#prueba 1 
estimated_modelh = esth.estimate( scoring_method=scoring_method, max_indegree=4,max_iter=int(1e4))
#primer blacklist
#blacklist =
blacklistT = [("thal","num") , ("oldpeak","num"), ("slope","num"),("ca","num"), ("restecg","num"),('ca', 'age'),('oldpeak', 'slope'),( 'slope','oldpeak'),('thal', 'sex'), ("num","sex"), ("sex", "thal"),('sex', 'exang'),('exang','sex')]
# #('cp', 'sex'), ('trestbps', 'sex'), ('chol', 'sex'), ('fbs', 'sex'), ('restecg', 'sex'), ('exang', 'sex'), ('oldpeak', 'sex'), ('slope', 'sex'), ('ca', 'sex'), ('thal', 'sex'), ('num', 'sex'),('age', 'exang'), ('sex', 'exang'), ('cp', 'exang'), ('trestbps', 'exang'), ('chol', 'exang'), ('fbs', 'exang'), ('restecg', 'exang'), ('oldpeak', 'exang'), ('slope', 'exang'), ('ca', 'exang'), ('thal', 'exang'), ('num', 'exang')
estimated_modelh = esth.estimate( scoring_method=scoring_method, max_indegree=4, black_list= blacklistT,max_iter=int(1e4))
print(estimated_modelh) 
print(estimated_modelh.nodes()) 
print(estimated_modelh.edges()) 

print(scoring_method.score(estimated_modelh))