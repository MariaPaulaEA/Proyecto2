#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:45:24 2023

@author: paulaescobar
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from pgmpy.readwrite import XMLBIFReader

# Read model from XML BIF file 
reader = XMLBIFReader("model.xml")
modelo = reader.get_model()

# Print model 
print(modelo)

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo.check_model()

for i in modelo.nodes():
    print(modelo.get_cpds(i))

# Infering the posterior probability
from pgmpy.inference import VariableElimination

infer = VariableElimination(modelo)
posterior_p = infer.query(["chol"], evidence={"age": "60_79", "sex": "1_0"})
print(posterior_p)

print(modelo.get_independencies())
