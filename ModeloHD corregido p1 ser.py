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

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFWriter

na_values = ["?"] 

df=pd.read_csv("/Users/paulaescobar/Documents/ACTD/Proyecto1/processed.cleveland.data.csv", sep = ",", na_values = na_values, header = None)
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
    print(model.get_cpds(i))


# check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
model.check_model()
print(model.nodes())
print(model.edges())

# write model to a BIF file 
writer = BIFWriter(model)
writer.write_bif(filename='modelo.bif')

# write model to a XML file 
from pgmpy.readwrite import XMLBIFWriter
writer = XMLBIFWriter(model)
writer.write_xmlbif('model.xml')

#Create SQL
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2

engine = create_engine('postgresql://postgres:Proyecto2023@proyecto2.cf0fdevvpqbv.us-east-1.rds.amazonaws.com/postgres', echo=False)

df.to_sql('tabla', con=engine, if_exists='replace', index=False)

#with engine.connect() as conn:
    #conn.execute(text("SELECT * FROM tabla")).fetchall()

#conn = engine.connect()

sql_text = text("""
                SELECT * FROM tabla
""")
df=pd.read_sql(sql_text, engine)

#1ra consulta

sql_text = text("""
                SELECT * FROM tabla
                WHERE age = :age
""")

with engine.connect() as conn:
    result = conn.execute(sql_text, age = '29-39')
    dff = pd.DataFrame(result)
    for row in result:
        print(row)

#2da consulta

sql_query = text("""
                SELECT * FROM tabla
                WHERE age = :age
                    AND num = :num
""")

params = {'age': '29-39', 'num': 2}

with engine.connect() as conn:
    result = conn.execute(sql_query, params)
    dff = pd.DataFrame(result)
    for row in result:
        print(row)


#3ra consulta

import sqlalchemy as db
from sqlalchemy import func

#Create Metadata object

meta_data = db.MetaData(bind=engine)
db.MetaData.reflect(meta_data)

#Get table from Metadata object

tabla = meta_data.tables['tabla']

#Query

query = db.select([tabla.c.age, tabla.c.num, func.count(tabla.c.num)]).order_by(tabla.c.age, tabla.c.num).group_by(tabla.c.age, tabla.c.num)
        
result = engine.execute(query).fetchall()

import plotly.express as px

dff = pd.DataFrame(result)
dff.columns = (["age", 'num', 'count'])

fig = px.scatter(dff, y="num", x="age", size="count")
 
fig.show()









