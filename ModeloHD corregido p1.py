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

df=pd.read_csv("C:/Users/ricky\Downloads/base/base/processed.cleveland.data.csv", sep = ",")
df.columns
df.columns = (["age", "sex", "cp", "trestbps", "chol", "fbs",
              "restecg", "thalach", "exang", "oldpeak", "slope",
              "ca", "thal", "num"])
df= df.drop('thalach', axis=1) 
df.head()


df= df.dropna()



model = BayesianNetwork ([( "age" , "fbs" ),
                          ( "age" , "chol" ),
                          ( "age", "trestbps" ),
                          ( "sex", "chol" ),
                          ( "chol" , "trestbps" ),
                          ( "fbs" , "trestbps" ),
                          ("trestbps", "cp"),
                          ( "exang" , "cp" ),
                          ( "exang" , "num" ),
                          ( "exang" , "thal" ),
                          ( "num" , "thal" ),
                          ( "num" , "ca" ),
                          ("num", "restecg"),
                          (  "num", "oldpeak" ),
                          ( "num", "slope"  ),
])



from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder as le
# EDAD
df["age"] = pd.cut(df['age'], bins=4, labels=['29-39', '40-49', '50-59', '60-79'])

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
emv = MaximumLikelihoodEstimator( model = model , data = df )
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
model.fit(data=df , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )

