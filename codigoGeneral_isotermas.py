from funciones_isotermas import promediar_y_calcular_qe, graficos_y_modelos, grafico_qe_vs_conc, estad_por_grupos

import pandas as pd

 #Subir archivo de cinetica
 
archivo = "isoterma FeS.txt"

"""Lee el archivo como un DataFrame:
 lo guarda en la variable "archivo"
 lo separa por tabulador e indica que el separador es decimal 
 (lo cambia por punto)
"""
df = pd.read_csv(archivo, sep="\t", decimal= ",")
print (" datos isotermas de adsorción \n")
print (df)

# Mostrar columnas detectadas

print("\nColumnas detectadas:")   
print(df.columns)


# promedio de variables y  calculo de qe
df_promedio = promediar_y_calcular_qe(df)
 
#calculo estadistica descriptiva 
estadistica= estad_por_grupos(df)

 
# grafico de datos experimentales (qe vs. t)
grafico= grafico_qe_vs_conc()

# evaluación del mejor ajuste
grafica_ajustes = graficos_y_modelos ()



