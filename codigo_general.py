
from funciones import promediar_y_calcular_qt, estad_por_grupos, grafico_qe_vs_t, ajustes_resultados

import pandas as pd

 #Subir archivo de cinetica
 
archivo = "cinetica-Fes.txt"

"""Lee el archivo como un DataFrame:
 lo guarda en la variable "archivo"
 lo separa por tabulador e indica que el separador es decimal 
 (lo cambia por punto)
"""
df = pd.read_csv(archivo, sep="\t", decimal= ",")
print (" datos cinética \n")
print (df)

# Mostrar columnas detectadas

print("\nColumnas detectadas:")   
print(df.columns)


# promedio de variables y calculo de qe
df_promedio = promediar_y_calcular_qt(df)

# calculo de estadística descriptiva
estadistica_descriptiva= estad_por_grupos (df)

# datos experimentales (qe vs. t)
grafico= grafico_qe_vs_t()

# evaluación del mejor ajuste
grafica_ajustes = ajustes_resultados ()

