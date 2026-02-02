#importar modulos 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import t

# Función para agrupar por tiempo, promediar Ci y Ce, y calcular qe

def promediar_y_calcular_qe(df):
    """ descripción: funcion calcular el promedio de Ci, Ce, m y el qe.
        y la estadística descriptiva: media, desvio estandar, n, error y coeficiente de confianza para qe al 95%
        Argumentos: 
                 promedios: promedio de cada una de las variables (Ce y m) y  V es constante
        returns: 
                nos da el valor de cada parámetro promediado y de qe-> qe= ((Ci-Ce)*V)/m  y
                las estadisticas (t, media_qe, std_qe, n, error y qe95) y realia la grafica de los datos 
                experimentales.      
    """
    # Promedio por grupo (Ci)
    
    global promedios
    promedios = df.groupby('Ci', as_index=False)[[ "Ce", 'm']].mean()

    # Tomar V correspondientes al primer valor por concentración
    global vm
    vm = df.groupby('Ci', as_index=False)[['V']].first()

    # Unir ambos dataframes
    global df_promedio
    df_promedio = pd.merge(promedios, vm, on='Ci') 

    # Calcular qe
    df_promedio['qe'] = (df_promedio['Ci'] - df_promedio['Ce']) * df_promedio['V'] / df_promedio['m']

    # Calcular qe individual para desviaciones
    
    df['qdsv'] = (df['Ci'] - df['Ce']) * df['V'] / df['m']

    # Mostrar el resultado final
    print("\nDatos promediados y qe calculado:")
    print(df_promedio)
    
def estad_por_grupos(df):
    #=== CÁLCULO DE ESTADÍSTICAS POR GRUPO ===
    global stats_df
    stats_df = df.groupby('Ci').agg(
      media_qe=('qdsv', 'mean'),
      std_qe=('qdsv', 'std'),
      n=('qdsv', 'count')).reset_index()
    
    stats_df['error-st'] = stats_df['std_qe'] / np.sqrt(stats_df['n'])  # error estándar
    stats_df['marg-error'] = stats_df['error-st'] * t.ppf(0.975, stats_df['n'])  # intervalo de confianza 95%

    print("\nEstadísticas por grupo:")
    print(stats_df)

def grafico_qe_vs_conc():
    # ========== GRÁFICO 2: qe vs Tiempo CON LÍNEA DE TENDENCIA AUTOMÁTICA ==========
   
    x = df_promedio['Ce']
    y = df_promedio['qe']
    global yerr
    yerr = stats_df['marg-error']

    # Ajuste automático: grados 1 a 4
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 0
    x_suave = np.linspace(min(x), max(x), 60)

    for grado in range(1, 4):
        coef = np.polyfit(x, y, deg=grado)
        modelo = np.poly1d(coef)
        y_pred = modelo(x)
        r2 = r2_score(y, y_pred)
        if r2 > mejor_r2:
           mejor_r2 = r2
           mejor_modelo = modelo
           mejor_grado = grado

     #=== GRAFICAR CON BARRAS DE ERROR ===
    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='#9c9c9c', lolims=0,
             label='Datos experimentales (IC 95%)')
    
 #Puntos experimentales
    plt.scatter(x, y, color='green', label='isoterma experimental')

# Curva ajustada
    plt.plot(x_suave, mejor_modelo(x_suave), ':', color='cornflowerblue',
         label=f'Ajuste grado {mejor_grado} (R² = {mejor_r2:.3f})')

    plt.xlabel('Ce (mg/L)')
    plt.ylabel('qe (mg/g)')
    plt.title('Isoterma de adsorción con ajuste polinomial')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def graficos_y_modelos():
   qe=df_promedio['qe']
   Ce=df_promedio['Ce']
   
   
    # Función Langmuir
   def langmuir_model(Ce, qmax, KL):
    return (qmax * KL * df_promedio['Ce']) / (1 + KL * df_promedio['Ce'])

#Funcion Temkin
   def temkin_model(Ce, At, Bt):
    return (Bt * np.log( At* df_promedio['Ce']))

#Funcion Freundlich
   def freund_model(Ce, Kf, nf):
    return (Kf* np.power(df_promedio['Ce'], (1/nf)))


# === AJUSTE LANGMUIR ===
   params_l, _ = curve_fit(langmuir_model, df_promedio['Ce'], qe, bounds=(0, np.inf))
   qmax_fit, KL_fit = params_l
   qe_ajustado_l = langmuir_model(df_promedio['Ce'], qmax_fit, KL_fit)
   r2_l = r2_score(qe, qe_ajustado_l)

# === AJUSTE TEMKIN ===
   params_t, _ = curve_fit(temkin_model, df_promedio['Ce'], qe, bounds=(0, np.inf))
   At_fit, Bt_fit = params_t
   qe_ajustado_t = temkin_model(df_promedio['Ce'], At_fit, Bt_fit)
   r2_t = r2_score(qe, qe_ajustado_t)

# === AJUSTE FREUNDLICH ===
   params_f, _ = curve_fit(freund_model, df_promedio['Ce'], qe, bounds=(0, np.inf))
   Kf_fit, nf_fit = params_f
   qe_ajustado_f = freund_model(df_promedio['Ce'], Kf_fit, nf_fit)
   r2_f = r2_score(qe, qe_ajustado_f)

# === EVALUAR EL MEJOR AJUSTE ===
   r2_dict = {
    'Langmuir': r2_l,
    'Temkin': r2_t,
    'Freundlich': r2_f
     } 
   mejor_modelo = max(r2_dict, key=r2_dict.get) + " 👑"

# === MOSTRAR PARÁMETROS ===
   print(f"\n🔹 Langmuir:")
   print(f"qmax = {qmax_fit:.4f}")
   print(f"KL = {KL_fit:.4f}")
   print(f"R² = {r2_l:.4f}")

   print(f"\n🔹 Temkin:")
   print(f"At = {At_fit:.4f}")
   print(f"Bt = {Bt_fit:.4f}")
   print(f"R² = {r2_t:.4f}")

   print(f"\n🔹 Freundlich:")
   print(f"Kf = {Kf_fit:.4f}")
   print(f"nf = {nf_fit:.4f}")
   print(f"1/nf = {(1/nf_fit):.4f}")
   print(f"R² = {r2_f:.4f}")

   print(f"\n Mejor modelo de ajuste: {mejor_modelo}")

# === CURVAS SUAVES PARA GRAFICAR ===
   Ce_suave = np.linspace(Ce.min(), Ce.max(), 5)
   qe_suave_l = langmuir_model(Ce_suave, qmax_fit, KL_fit)
   qe_suave_t = temkin_model(Ce_suave, At_fit, Bt_fit)
   qe_suave_f = temkin_model(Ce_suave, Kf_fit, nf_fit)

# === GRAFICAR ===
   plt.figure(figsize=(7, 5))
   #=== GRAFICAR CON BARRAS DE ERROR ===
   plt.errorbar(Ce, qe, yerr=yerr, fmt='o', capsize=5, color="#0c0c0c", lolims=0,
             label='Datos experimentales (IC 95%)')
   plt.scatter(Ce, qe, label='Datos experimentales', color='black')
   
   
    
   plt.plot(Ce, qe_suave_l, label=f'Langmuir (R²={r2_l:.3f})', color='red')
   plt.plot(Ce, qe_suave_t, label=f'Temkin (R²={r2_t:.3f})', color='green')
   plt.plot(Ce, qe_suave_f, label=f'Freundlich (R²={r2_f:.3f})', color='blue')
   plt.xlabel('Ce (mg/L)')
   plt.ylabel('qe (mg/g)')
   plt.title('Ajuste de Isotermas: Langmuir vs Temkin vs Freundlich')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   plt.show()
