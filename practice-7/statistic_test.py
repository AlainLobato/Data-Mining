import pandas as pd
from scipy.stats import kruskal, f_oneway

# Leemos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Convertimos la columna de fecha a formato datetime para poder extraer fácilmente el mes y el año, lo que nos permitirá hacer comparaciones temporales.
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Creamos una función para interpretar los resultados de las pruebas estadísticas.
# Usamos un valor de p < 0.05 porque es el valor comúnmente aceptado para rechazar la hipótesis nula,
# para considerar que existen diferencias significativas entre los grupos que estamos comparando.
def interpretacion(p):
    if p < 0.05:
        return "Hay diferencias significativas entre los grupos.\n"
    else:
        return "No hay diferencias significativas entre los grupos.\n"

def print_results(test_name, stat, p):
    print(f"{test_name}")
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    print(interpretacion(p))

# --- PM2.5 entre diferentes estaciones ---
# Agrupamos los datos por estación y preparamos los valores para las pruebas estadísticas.
groups_station = [g['PM2.5'].dropna().values for _, g in df.groupby('station_name')]

# Usamos la prueba de Kruskal-Wallis para ver si existen diferencias significativas entre las estaciones,
# ya que es una prueba no paramétrica adecuada cuando no podemos asumir normalidad en los datos.
stat, p = kruskal(*groups_station)
print_results("\nKruskal-Wallis: PM2.5 por estación", stat, p)

# También usamos ANOVA para comparar los promedios de PM2.5 entre estaciones,
# asumiendo que los datos pueden ser aproximadamente normales.
stat, p = f_oneway(*groups_station)
print_results("ANOVA: PM2.5 por estación", stat, p)


# --- CO entre los diferentes meses ---
# Agrupamos los datos por mes para analizar posibles patrones estacionales.
groups_month = [g['CO'].dropna().values for _, g in df.groupby('month')]

# Usamos Kruskal-Wallis para ver si hay diferencias en los niveles de CO entre meses,
# lo cual puede indicar variaciones estacionales en la calidad del aire.
stat, p = kruskal(*groups_month)
print_results("\nKruskal-Wallis: CO por mes", stat, p)

# Aplicamos ANOVA para comparar los promedios de CO entre meses.
stat, p = f_oneway(*groups_month)
print_results("ANOVA: CO por mes", stat, p)


# --- O3 entre diferentes años ---
# Agrupamos los datos por año para observar posibles tendencias a largo plazo.
groups_year = [g['O3'].dropna().values for _, g in df.groupby('year')]

# Usamos Kruskal-Wallis para ver si hay diferencias en los niveles de O3 entre años,
# lo que puede mostrar tendencias o cambios a lo largo del tiempo.
stat, p = kruskal(*groups_year)
print_results("\nKruskal-Wallis: O3 por año", stat, p)

# Aplicamos ANOVA para comparar los promedios de O3 entre años.
stat, p = f_oneway(*groups_year)
print_results("ANOVA: O3 por año", stat, p)


# Los resultados obtenidos fueron:

# PM2.5 por estación

# Kruskal-Wallis:
# Statistics=438.344, p=0.00000
# Hay diferencias significativas entre los grupos.

# ANOVA:
# Statistics=35.200, p=0.00000
# Hay diferencias significativas entre los grupos.

# Explicación:
# Tanto Kruskal-Wallis como ANOVA arrojaron p < 0.05.

# Esto indica que las concentraciones de PM2.5 no son iguales en todas las estaciones de Monterrey, lo cual tiene sentido porque la ubicación geográfica influye en los niveles de contaminación (zonas industriales, tráfico, áreas residenciales).


# CO por mes

# Kruskal-Wallis:
# Statistics=1378.933, p=0.00000
# Hay diferencias significativas entre los grupos.

# ANOVA:
# Statistics=130.934, p=0.00000
# Hay diferencias significativas entre los grupos.

# Explicación:

# Nuevamente p < 0.05 en ambas pruebas.

# Con esto podemos afirmar que las concentraciones de CO varían significativamente según el mes, es decir, hay una componente estacional (ejemplo: en invierno puede aumentar por calefacciones, en verano puede bajar por mayor dispersión del aire).


# O3 por año

# Kruskal-Wallis:
# Statistics=479.516, p=0.00000
# Hay diferencias significativas entre los grupos.

# ANOVA:
# Statistics=101.053, p=0.00000
# Hay diferencias significativas entre los grupos.

# También p < 0.05 en ambos casos.

# Esto significa que los niveles de ozono han cambiado con los años, probablemente reflejando tendencias de largo plazo en la calidad del aire (más emisiones o mejoras en políticas ambientales).
