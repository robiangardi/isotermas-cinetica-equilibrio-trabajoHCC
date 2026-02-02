#importar modulos 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import t

# Función para agrupar por tiempo, promediar Ci y Ce, y calcular qe

def promediar_y_calcular_qt(df):
    """ descripción: funcion calcular el promedio de Ci, Ce y m y el qe.
        
        Argumentos: 
                 promedios: promedio de cada una de las variables (Ci, Ce y m), V es constante, 
                 en función del tiempo
        returns: 
                nos da el valor de cada parámetro promediado y de qe-> qe= ((Ci-Ce)*V)/m      
    """
    # Promedio de Ci, m y Ce por tiempo

    global promedios
    promedios = df.groupby('t', as_index=False)[['Ci', 'Ce','m']].mean()

    # Tomar V correspondientes al primer valor por tiempo
    vm = df.groupby('t', as_index=False)[['V']].first()

    # Unir ambos dataframes
    global df_promedio
    df_promedio = pd.merge(promedios, vm, on='t') 

    # Calcular qe
    df_promedio['qt'] = (df_promedio['Ci'] - df_promedio['Ce']) * df_promedio['V'] / df_promedio['m']

    # Calcular qe individual para desviaciones
    df['qdsv'] = (df['Ci'] - df['Ce']) * df['V'] / df['m']

    # Mostrar el resultado final
    print("\nDatos promediados y qe calculado:")
    print(df_promedio)

def estad_por_grupos(df):  
    """
 función que describe la estadística descriptiva

 argumentos: media, desvio estandar, n, error y coeficiente de confianza para qe al 95%

 return: devuelve las estadísticas de los datos agrupados por tiempoas estadisticas (t, media_qe, std_qe, n, error y qe95) 
    """ 
#=== CÁLCULO DE ESTADÍSTICAS POR GRUPO ===
    global stats_df
    stats_df = df.groupby('t').agg(
    media_qt=('qdsv', 'mean'),
    std_qt=('qdsv', 'std'),
    n=('qdsv', 'count')).reset_index()

    stats_df['error-st'] = stats_df['std_qt'] / np.sqrt(stats_df['n'])  # error estándar
    stats_df['marg-error'] = stats_df['error-st'] * t.ppf(0.95, stats_df['n'])  # intervalo de confianza 95%

    print("\nEstadísticas por grupo:")
    print(stats_df)

def grafico_qe_vs_t():
    # ========== GRÁFICO 2: qe vs Tiempo CON LÍNEA DE TENDENCIA AUTOMÁTICA ==========

    x2 = df_promedio['t']
    y2 = df_promedio['qt']
    global yerr
    yerr = stats_df['marg-error']

    # Ajuste automático: grados 1 a 4
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_grado = 0
    x2_suave = np.linspace(min(x2), max(x2), 50)

    for grado in range(1, 5):
        coef = np.polyfit(x2, y2, deg=grado)
        modelo = np.poly1d(coef)
        y2_pred = modelo(x2)
        r2 = r2_score(y2, y2_pred)
        if r2 > mejor_r2:
           mejor_r2 = r2
           mejor_modelo = modelo
           mejor_grado = grado

# Graficar

    plt.figure(figsize=(6, 4))

# Error bars con control de límites inferiores
    plt.errorbar(
    x2, y2, yerr=yerr,
    fmt='o', capsize=5,
    color='#9c9c9c',
    lolims=0,
    label='Datos experimentales (IC 95%)'
)

# Puntos experimentales
    plt.scatter(x2, y2, color='green', label='Cinetica experimental')

# Curva ajustada
    plt.plot(x2_suave, mejor_modelo(x2_suave), ':', color='#88E788',
         label=f'Ajuste grado {mejor_grado} (R² = {mejor_r2:.3f})')

# Ejes y formato
    plt.xlabel('Tiempo (min)')
    plt.ylabel('qt (mg/g)')
    plt.title('Curva cinética: qe vs tiempo')
    plt.ylim(bottom=0)  # Limita visualmente a partir de 0 en el eje Y
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def ajustes_resultados():
    """función que calcula los modelos teóricos de pseudo primer orden, pseudo segundo orden y Temkin
       y evalúa a que modelo ajustan mejor los datos experimentales.
       argumentos: las funciones de cada modelo ingresa variables de tiempo, qe y constantes especificas de
       cada función
       return: muestra la grafica de los datos experimentales y de los tres modelos teoricos a los que se 
       ajustan los datos.
    """
    #promediar_y_calcular_qe(df_promedio)
    t = df_promedio['t']
    qt = df_promedio['qt']

    
# === DEFINICIÓN DE MODELOS ===

# Pseudo primer orden
    def pseudoPrimerOrden(t, qt, K1):
      return qt * (1 - np.exp(-K1 * t))

# Pseudo segundo orden
    def pseudoSegundoOrden(t, qt, K2):
       return (qt**2 * K2 * t) / (1 + qt * K2 * t)

# Elovich
    def elovich(t, a, b):
      return (1 / b) * np.log(1 + a * b * t)

   #
# === AJUSTES ===

# PFO
    params_pfo, _ = curve_fit(pseudoPrimerOrden, t, qt, bounds=(0, np.inf))
    qe_fit_pfo, K1_fit = params_pfo
    qt_ajustado_pfo = pseudoPrimerOrden(t, qe_fit_pfo, K1_fit)
    r2_pfo = r2_score(qt, qt_ajustado_pfo)

# PSO
    params_pso, _ = curve_fit(pseudoSegundoOrden, t, qt, bounds=(0, np.inf))
    qe_fit_pso, K2_fit = params_pso
    qt_ajustado_pso = pseudoSegundoOrden(t, qe_fit_pso, K2_fit)
    r2_pso = r2_score(qt, qt_ajustado_pso)

# Elovich
    params_elovich, _ = curve_fit(elovich, t, qt, bounds=(0, np.inf))
    a_fit, b_fit = params_elovich
    qt_ajustado_elovich = elovich(t, a_fit, b_fit)
    r2_elovich = r2_score(qt, qt_ajustado_elovich)

# === IMPRESIÓN DE RESULTADOS ===

    print("\n🔹 Modelo pseudo primer orden:")
    print(f"qt = {qe_fit_pfo:.4f} mg/g, K1 = {K1_fit:.4f} 1/min, R² = {r2_pfo:.4f}")
    print("\n🔹 Modelo pseudo segundo orden:")
    print(f"qt = {qe_fit_pso:.4f} mg/g, K2 = {K2_fit:.4f} g/mg·min, R² = {r2_pso:.4f}")

    print("\n🔹 Modelo Elovich:")
    print(f"a = {a_fit:.4f}, b = {b_fit:.4f}, R² = {r2_elovich:.4f}")

# === EVALUAR EL MEJOR AJUSTE ===
    r2_dict = {
      ' Pseudo Primer Orden': r2_pfo,
       'Pseudo Segundo Orden': r2_pso,
         'Elovich': r2_elovich
          }
    mejor_modelo = max(r2_dict, key=r2_dict.get) + " 👑"
    print(f"\n Mejor modelo de ajuste: {mejor_modelo}")

# === GRAFICAR ===
    plt.figure(figsize=(8, 5))
    #Error bars con control de límites inferiores
    plt.errorbar(
    t, qt, yerr=yerr, fmt='o', capsize=5, color="#111010", lolims=0, label='Datos experimentales (IC 95%)'
)
    plt.scatter(t, qt, label='Datos experimentales', color='black')
    plt.plot(t, qt_ajustado_pfo, label=f'PFO (R²={r2_pfo:.3f})', color='green')
    plt.plot(t, qt_ajustado_pso, label=f'PSO (R²={r2_pso:.3f})', color='orange')
    plt.plot(t, qt_ajustado_elovich, label=f'Elovich (R²={r2_elovich:.3f})', color='cornflowerblue')

    plt.xlabel('Tiempo (min)')
    plt.ylabel('qt (mg/g)')
    plt.title('Comparación de modelos cinéticos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()