#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:13:23 2023

@author: paulaescobar
"""

import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd

from pgmpy.sampling import BayesianModelSampling

from pgmpy.readwrite import XMLBIFReader

from sqlalchemy import create_engine
from sqlalchemy import text
import sqlalchemy as db
from sqlalchemy import func
import psycopg2

# Read model from XML BIF file 
reader = XMLBIFReader("model.xml")
model = reader.get_model()

infer = VariableElimination(model)

#Conectarse a base de datos

engine = create_engine('postgresql://postgres:Proyecto2023@proyecto2.cf0fdevvpqbv.us-east-1.rds.amazonaws.com/postgres', echo=False)

sql_text = text("""
                SELECT * FROM tabla
""")
df=pd.read_sql(sql_text, engine)

#Create Metadata object

meta_data = db.MetaData()
meta_data.reflect(bind=engine)
        
#Get table from Metadata object
        
tabla = meta_data.tables['tabla']

#TABLERO DASH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    
    html.H1("Herramienta de apoyo de analítica de datos en el proceso de evaluación de pacientes y la toma de decisiones asociada",
            style={'text-align': 'center', 'color': 'white', 'backgroundColor': '#0b0347'}),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        
        dcc.Tab(label='Instrucciones', value='tab-1', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.Br(),
            
            html.H4("En esta herramienta encontrará información para el apoyo en el proceso de evaluación de pacientes de enfermedad cardiaca y la toma de decisiones asociada. La información parte de la base de datos de enfermedad cardiaca de la Clínica de Cleveland.",
                    style={'text-align': 'justify', 'backgroundColor': '#6a94c4'}),
            
            html.Br(),
            
            html.H4("La herramienta se divide en 4 secciones posibles para consulta:",
                    style={'text-align': 'justify', 'backgroundColor': '#cee5ed'}),
            
            html.H5("-Gráficas de interés para edad, sexo y angina inducida por ejercicio: Consulte gráficas que relacionan el diagnóstico de la enfermedad cardiaca con edad, sexo y angina inducida del paciente.", style={'backgroundColor': '#cee5ed'}),
            
            html.H5("-Gráficas de interés para exámenes médicos: Consulte gráficas que relacionan el diagnóstico de la enfermedad cardiaca con el resultado de distintos exámenes médicos.", style={'backgroundColor': '#cee5ed'}),
            
            html.H5("-Predicción de resultados de exámenes médicos: Consulte la probabilidad de obtener cierto resultado para un exámen médico de su elección, a partir de valores de edad, sexo y angina inducida.", style={'backgroundColor': '#cee5ed'}),
            
            html.H5("-Predicción de resultados de diagnóstico de enfermedad cardiaca: Consulte la probabilidad de obtener cierto diagnóstico de enfermedad cardiaca, a partir de valores de edad, sexo, angina inducida, colesterol, presión arterial en reposo y azúcar en la sangre en ayunas.", style={'backgroundColor': '#cee5ed'}),
            
            ]),
        
        dcc.Tab(label='Gráficas de interés: Edad, Sexo y Angina', value='tab-2', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2("Gráficas de interés para edad, sexo y angina inducida por ejercicio",
            style={'backgroundColor': '#6a94c4'}),
            
            html.H6("(Sitúe el cursor sobre los puntos para conocer más información)"),
            
            html.H4("Diagrama de burbuja para diagnóstico de la enfermedad respecto a la edad:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph3'),
            
            html.H4("Diagrama de burbuja para diagnóstico de la enfermedad respecto a sexo del paciente:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph4'),
            
            html.H4("Diagrama de burbuja para diagnóstico de la enfermedad respecto a la angina inducida por ejercicio del paciente:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph5'),
            
             ]),
        
        dcc.Tab(label='Gráficas de interés: Exámenes', value='tab-3', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2("Gráficas de interés para exámenes médicos",
            style={'backgroundColor': '#6a94c4'}),
            
            html.H6("(Sitúe el cursor sobre los puntos para conocer más información)"),
            
            html.H4("Diagnóstico de la enfermedad en relación con Estado del corazón según prueba Thallium:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph6'),
            
            html.H4("Diagnóstico de la enfermedad en relación con Tipo de dolor en el pecho:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph7'),
            
            html.H4("Diagnóstico de la enfermedad en relación con Resultados de electrocardiograma en reposo:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph8'),
            
            html.H4("Diagnóstico de la enfermedad en relación con Depresión del ST inducida por el ejercicio en relación con el descanso:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph9'),
            
            html.H4("Diagnóstico de la enfermedad en relación con Pendiente del segmento ST:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph10'),
            
            html.H4("Diagnóstico de la enfermedad en relación con Número de vasos principales coloreados por fluoroscopia:", style={'backgroundColor': '#cee5ed'}),
            
            dcc.Graph(id='graph11'),
            
            ]),
        
        dcc.Tab(label='Predicción: Exámenes', value='tab-4', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2("Estimación de probabilidades del resultado del paciente para un exámen médico determinado",
            style={'backgroundColor': '#6a94c4'}),

            html.Div(children=[
            
                html.H5("Seleccione el exámen de interés:"),
            
                html.Div(
                    className="Seleccion", children=[
                        dcc.Dropdown(
                            id="dropdownExamen",
                            options=['Estado del corazón según prueba Thallium', 
                                     'Tipo de dolor en el pecho',
                                     'Resultados de electrocardiograma en reposo',
                                     'Depresión del ST inducida por el ejercicio en relación con el descanso',
                                     'Pendiente del segmento ST',
                                     'Número de vasos principales coloreados por fluoroscopia'],
                            value="Estado del corazón según prueba Thallium",
                            clearable=False,
                            ),
                    ],),
                
                html.Br(),
                
                html.Div(children=[
                    html.Div(children=[
                            html.Label("Seleccione el rango de edad:", htmlFor = "dropdownEdad"),
                            dcc.Dropdown(
                                id="dropdownEdad",
                                options=['29-39', '40-49', '50-59', '60-79'],
                                value='29-39',
                                clearable=False,
                                ),
                            ],style=dict(width='33%')),
                    
                    html.Div(children=[
                            html.Label("Seleccione el sexo:", htmlFor = "dropdownSexo"),
                            dcc.Dropdown(
                                id="dropdownSexo",
                                options=["Femenino","Masculino"],
                                value="Femenino",
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                    
                     html.Div(children=[
                            html.Label("Seleccione angina inducida por ejercicio:", htmlFor = "dropdownAngina"),
                            dcc.Dropdown(
                                id="dropdownAngina",
                                options=['No tiene angina inducida por ejercicio','Tiene angina inducida por ejercicio'],
                                value='No tiene angina inducida por ejercicio',
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                     ],style=dict(display='flex')),
                
                html.Br(),
                
                html.H4("Distribución de probabilidad para resultados del paciente según edad, sexo y angina para el exámen seleccionado:", style={'backgroundColor': '#cee5ed'}),
                
                dcc.Graph(id='graph'),
                
                ]),
            ]),
        
        dcc.Tab(label='Predicción: Diagnóstico', value='tab-5', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2("Estimación de probabilidades del diagnóstico del paciente respecto a la enfermedad cardiaca",
                        style={'backgroundColor': '#6a94c4'}),

            html.Div(children=[
            
                html.Div(children=[
                    html.Div(children=[
                            html.Label("Seleccione el rango de edad:", htmlFor = "dropdownEdad"),
                            dcc.Dropdown(
                                id="dropdownEdad2",
                                options=['29-39', '40-49', '50-59', '60-79'],
                                value='29-39',
                                clearable=False,
                                ),
                            ],style=dict(width='33%')),
                    
                    html.Div(children=[
                            html.Label("Seleccione el sexo:", htmlFor = "dropdownSexo"),
                            dcc.Dropdown(
                                id="dropdownSexo2",
                                options=["Femenino","Masculino"],
                                value="Femenino",
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                    
                     html.Div(children=[
                            html.Label("Seleccione angina inducida por ejercicio:", htmlFor = "dropdownAngina"),
                            dcc.Dropdown(
                                id="dropdownAngina2",
                                options=['No tiene angina inducida por ejercicio','Tiene angina inducida por ejercicio'],
                                value='No tiene angina inducida por ejercicio',
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                     ],style=dict(display='flex')),
                
                html.Br(),
                
                html.Div(children=[
                    html.Div(children=[
                            html.Label("Seleccione el rango de colesterol en mg/dl:", htmlFor = "dropdownCol"),
                            dcc.Dropdown(
                                id="dropdownCol",
                                options=['< 200', '200-239', '240 >='],
                                value='< 200',
                                clearable=False,
                                ),
                            ],style=dict(width='33%')),
                    
                    html.Div(children=[
                            html.Label("Seleccione el rango de presión arterial en reposo en mm Hg:", htmlFor = "dropdownPre"),
                            dcc.Dropdown(
                                id="dropdownPre",
                                options=['80-120', '120-129', '130-139', '140-180', '180 >='],
                                value="80-120",
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                    
                     html.Div(children=[
                            html.Label("Seleccione el nivel de azúcar en la sangre en ayunas:",
                                       htmlFor = "dropdownAzu"),
                            dcc.Dropdown(
                                id="dropdownAzu",
                                options=['Es menor a 120 mg/dl','Es mayor a 120 mg/dl'],
                                value='Es mayor a 120 mg/dl',
                                clearable=False,
                                ),
                        ],style=dict(width='33%')),
                     ],style=dict(display='flex')),
                
                html.H4("Distribución de probabilidad para diagnóstico del paciente según edad, sexo, angina, niveles de colesterol, presión arterial y azúcar en ayunas:", style={'backgroundColor': '#cee5ed'}),
                
                html.H6("*Si la combinación de variables del paciente no cuenta con evidencias previas, la gráfica aparece en blanco*"),
                
                dcc.Graph(id='graph2'),
            
                ]),
            
             ]),

    ], style=tabs_styles),

])

@app.callback(Output('graph', 'figure'),
              Output('graph2', 'figure'),
              Output('graph3', 'figure'),
              Output('graph4', 'figure'),
              Output('graph5', 'figure'),
              Output('graph6', 'figure'),
              Output('graph7', 'figure'),
              Output('graph8', 'figure'),
              Output('graph9', 'figure'),
              Output('graph10', 'figure'),
              Output('graph11', 'figure'),
              Input('tabs', 'value'),
              Input('dropdownExamen', 'value'),
              Input('dropdownEdad', 'value'),
              Input('dropdownSexo', 'value'),
              Input('dropdownAngina', 'value'),
              Input('dropdownEdad2', 'value'),
              Input('dropdownSexo2', 'value'),
              Input('dropdownAngina2', 'value'),
              Input('dropdownCol', 'value'),
              Input('dropdownPre', 'value'),
              Input('dropdownAzu', 'value'))

def update_output_div(tab, selected_Examen, selected_Age, selected_Sex, selected_Angina, selected_Age2, selected_Sex2, selected_Angina2, selected_Col, selected_Pre, selected_Azu):
    if tab == 'tab-4':
      
        sexo = '0_0'
        angina = '0_0'
        col = 'saludable'
        pre = 'normal'
        azucar = '0_0'
        edad = '29_39'
        
        #Edad
        if selected_Age == "29-39":
            edad = '29_39'
        elif selected_Age == "40-49":
            edad = '40_49'
        elif selected_Age == "50-59":
            edad = '50_59'
        elif selected_Age == "60-79":
            edad = '60_79'
      
        #Sexo  
        if selected_Sex == "Fememino":
            sexo = '0_0'
        elif selected_Sex == "Masculino":
            sexo = '1_0'
            
        #Angina
        if selected_Angina == "No tiene angina inducida por ejercicio":
            angina = '0_0'
        elif selected_Angina == "Tiene angina inducida por ejercicio":
            angina = '1_0'
            
        #Examen
        
        if selected_Examen == "Estado del corazón según prueba Thallium":
            posterior_p = infer.query(["thal"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = (["Normal", 'Defecto fijo', 'Defecto reversible'])
        elif selected_Examen == "Tipo de dolor en el pecho":
            posterior_p = infer.query(["cp"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = (["Angina típica", 'Angina atípica', 'Dolor no-anginal', 'Asintomático'])
        elif selected_Examen == "Resultados de electrocardiograma en reposo":
            posterior_p = infer.query(["restecg"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = (["Normal", 'Anormalidad de onda ST-T', 'Hipertropía ventricular izquierda probable o definitiva'])
        elif selected_Examen == "Depresión del ST inducida por el ejercicio en relación con el descanso":
            posterior_p = infer.query(["oldpeak"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = (["Baja", 'Normal', 'Terrible'])
        elif selected_Examen == "Pendiente del segmento ST":
            posterior_p = infer.query(["slope"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = (["Ascenso", 'Plano', 'Descenso'])
        elif selected_Examen == "Número de vasos principales coloreados por fluoroscopia":
            posterior_p = infer.query(["ca"], evidence={"age": edad, "sex": sexo, 'exang': angina})
            estados = ([0, 1, 2, 3])
        
        dff = pd.DataFrame(posterior_p.values)
        dff.columns = (["Probabilidad"])
        dff['Estados'] = estados
        
        fig = px.bar(dff, x = "Estados", y = "Probabilidad",
                         labels={
                         "Estados": "Resultado",
                         }
                        )
        
        if selected_Examen == "Número de vasos principales coloreados por fluoroscopia":
            fig.update_xaxes(dtick = 1)
        
        return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    elif tab == 'tab-2':
        
        #Query edad
        
        #query = db.select(tabla.c.age, tabla.c.num, db.func.count(tabla.c.num)).order_by(tabla.c.age, tabla.c.num).group_by(tabla.c.age, tabla.c.num)
        #result = engine.connect().execute(query).fetchall()
        
        query = db.select([tabla.c.age, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.age, tabla.c.num).group_by(tabla.c.age, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Edad", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        fig = px.scatter(dff, x="Edad", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query sexo
        
        query = db.select([tabla.c.sex, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.sex, tabla.c.num).group_by(tabla.c.sex, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Sexo", 'Diagnóstico', 'Cantidad'])
        dff['Sexo'] = dff['Sexo'].replace([0, 1], ['Femenino', 'Masculino'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        
        fig2 = px.scatter(dff, x="Sexo", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        fig2.update_layout(width=800)
        
        #Query angina
        
        query = db.select([tabla.c.exang, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.exang, tabla.c.num).group_by(tabla.c.exang, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Angina inducida por el ejercicio", 'Diagnóstico', 'Cantidad'])
        dff['Angina inducida por el ejercicio'] = dff['Angina inducida por el ejercicio'].replace([0, 1], ['No tiene angina inducida', 'Sí tiene angina inducida'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        
        fig3 = px.scatter(dff, x="Angina inducida por el ejercicio", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        fig3.update_layout(width=800)
        
        return dash.no_update, dash.no_update, fig, fig2, fig3, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    elif tab == 'tab-1':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif tab == 'tab-5':
        
        sexo = '0_0'
        angina = '0_0'
        col = 'saludable'
        pre = 'normal'
        azucar = '0_0'
        edad = '29_39'
        
        #Edad
        if selected_Age2 == "29-39":
            edad = '29_39'
        elif selected_Age2 == "40-49":
            edad = '40_49'
        elif selected_Age2 == "50-59":
            edad = '50_59'
        elif selected_Age2 == "60-79":
            edad = '60_79'
      
        #Sexo  
        if selected_Sex2 == "Fememino":
            sexo = '0_0'
        elif selected_Sex2 == "Masculino":
            sexo = '1_0'
            
        #Angina
        if selected_Angina2 == "No tiene angina inducida por ejercicio":
            angina = '0_0'
        elif selected_Angina2 == "Tiene angina inducida por ejercicio":
            angina = '1_0'
        
        #Colesterol
        if selected_Col == "< 200":
            col = 'saludable'
        elif selected_Col == "200-239":
            col = 'riesgoso'
        elif selected_Col == "240 >=":
            col = 'peligroso'
            
        #Presión
        if selected_Pre == '80-120':
            pre = 'normal'
        elif selected_Pre == '120-129':
            pre = 'elevada'
        elif selected_Pre == '130-139':
            pre = 'hiptertensi_n_nivel_1'
        elif selected_Pre == '140-180':
            pre = 'hiptertensi_n_nivel_2'
        elif selected_Pre == '180 >=':
            pre = 'crisis_hipertensi_n'
        
        #Azucar
        if selected_Azu == "Es mayor a 120 mg/dl":
            azucar = '0_0'
        elif selected_Azu == "Es menor a 120 mg/dl":
            azucar = '1_0'
            
        posterior_num = infer.query(["num"], evidence={"age": edad, "sex": sexo, 'exang': angina, 'chol': col, 'trestbps': pre, 'fbs': azucar})
        dfff = pd.DataFrame(posterior_num.values)
        dfff.columns = (["Probabilidad"])
        dfff['Diagnóstico'] = ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4']
        
        fig2 = px.bar(dfff, x = "Diagnóstico", y = "Probabilidad")
        
        return dash.no_update, fig2, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif tab == 'tab-3':
        
        #Query Thallium
        
        query = db.select([
            tabla.c.thal, tabla.c.num, db.func.count(tabla.c.num)
            ]).order_by(tabla.c.thal, tabla.c.num).group_by(tabla.c.thal, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        dff['Resultado'] = dff['Resultado'].replace([3, 6, 7], ['Normal', 'Defecto fijo', 'Defecto reversible'])
        fig = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query Dolor pecho
        
        query = db.select([tabla.c.cp, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.cp, tabla.c.num).group_by(tabla.c.cp, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        dff['Resultado'] = dff['Resultado'].replace([1, 2, 3, 4], ['Angina típica', 'Angina atípica', 'Dolor no-anginal', 'Asintomático'])
        fig2 = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query Electrocardiograma
        
        query = db.select([tabla.c.restecg, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.restecg, tabla.c.num).group_by(tabla.c.restecg, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        dff['Resultado'] = dff['Resultado'].replace([0, 1, 2], ["Normal", 'Anormalidad de onda ST-T', 'Hipertropía ventricular izquierda probable o definitiva'])
        fig3 = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query Depresión ST
        
        query = db.select([tabla.c.oldpeak, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.oldpeak, tabla.c.num).group_by(tabla.c.oldpeak, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        dff['Resultado'] = dff['Resultado'].replace(['baja', 'normal', 'terrible'], ["Baja", 'Normal', 'Terrible'])
        fig4 = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query Pendiente ST
        
        query = db.select([tabla.c.slope, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.slope, tabla.c.num).group_by(tabla.c.slope, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        dff['Resultado'] = dff['Resultado'].replace([1, 2, 3], ["Ascenso", 'Plano', 'Descenso'])
        fig5 = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        
        #Query Vasos coloreados
        
        query = db.select([tabla.c.ca, tabla.c.num, db.func.count(tabla.c.num)]).order_by(tabla.c.ca, tabla.c.num).group_by(tabla.c.ca, tabla.c.num)
    
        result = engine.execute(query).fetchall()
        
        dff = pd.DataFrame(result)
        dff.columns = (["Resultado", 'Diagnóstico', 'Cantidad'])
        dff['Diagnóstico'] = dff['Diagnóstico'].replace([0, 1, 2, 3, 4], ['Saludable', 'Etapa 1', 'Etapa 2', 'Etapa 3', 'Etapa 4'])
        fig6 = px.scatter(dff, x="Resultado", y="Diagnóstico", size="Cantidad", color="Cantidad", color_continuous_scale="sunsetdark")
        fig6.update_xaxes(dtick = 1)
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run_server(debug=True)
