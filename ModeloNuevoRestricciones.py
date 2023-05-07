import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC 
import plotly.express as px
import pandas as pd
import itertools
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination 
import dash

from dash.dependencies import Input, Output

from pgmpy.sampling import BayesianModelSampling 


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
from pgmpy.estimators import PC
from pgmpy.sampling import BayesianModelSampling

from pgmpy.estimators import StructureEstimator

import pandas as pd  

from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)
"""from pgmpy.global_vars import SHOW_PROGRESS
class HillClimbSearch(StructureEstimator):
    def __init__(self, data, use_cache=True, **kwargs): 

        def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
                    tabu_list = set(tabu_list)
"""

                    
# Definir las restricciones de la lista blanca y negra
fixed_edges = [("age", "chol"),("age", "fbs"), ( "age", "trestbps"),  ("sex", "chol"), ("chol","trestbps"), ("fbs", "trestbps"),("exang" , "cp")]
blacklist = [("thal","num") , ("oldpeak","num"), ("slope","num"),("ca","num"), ("restecg","num")]
dff=pd.read_csv("C:\\Users\\ricky\\Downloads\\Dash\\datos_80.csv")
est =PC(data=dff)
# Crear el estimador de restricciones
"""est =PC(data=df, whitelist= whitelist, blacklist= blacklist )




# Establecer las restricciones de las listas en el estimador


est= WhitelistEstimator(est, whitelist)
est=BlacklistEstimator(est, blacklist)"""


"""# Establecer el número máximo de enlaces
est.max_reach = 3

# Estimar la estructura de la red bayesiana
structure = est.estimate()
"""
# estimar  la estructura 

estimated_model = est.estimate(variant="stable", max_cond_vars=4 )
print(estimated_model)
print(estimated_model.nodes())
print(estimated_model.edges())
