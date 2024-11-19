import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

# Dataset Medicamentos
df_medicamentos = pd.read_csv("dim_medicamento.csv")
df_medicamentos.head()

# Dataset Medicamentos Recetados
df_medicamentos_recetados = pd.read_csv("medicamentos_recetados.csv")
df_medicamentos_recetados.head()


# Merge de los dos datasets
df_merged = df_medicamentos_recetados.merge(
    df_medicamentos,
    how='left',
    left_on='key_medicamento',
    right_on='key_medicamento')[['codigo_formula', 'Nombre Generico']]
df_merged.head()

# Lista de medicamentos por cada codigo de formula se hace una fila que tiene una lista de los medicamentos
lista_medicamentos_formula = []
for codigo_formula in df_merged['codigo_formula'].unique():
    lista_medicamentos_formula.append(
        df_merged[df_merged['codigo_formula'] == codigo_formula]['Nombre Generico'].values.tolist())

rules = apriori(lista_medicamentos_formula, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
results = list(rules)
results_df = pd.DataFrame(results)
results_df.head(30)


# putting output into a pandas dataframe
def inspect(output):
    lhs = [tuple(result[2][0][0])[0] for result in output]
    rhs = [tuple(result[2][0][1])[0] for result in output]
    support = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))


# Otra forma de visualizar los resultados
output = list(results)
output_DataFrame = pd.DataFrame(inspect(results), columns=[
    'Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'
])
output_DataFrame.head(30)
