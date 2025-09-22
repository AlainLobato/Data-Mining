# Practica 6: Data Visualization

# Cargamos nuestras librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Extraer variables de tiempo, para poder separar por año y mes, debemos tener un datetime asi que lo generamos
df["datetime"] = pd.to_datetime(df["datetime"])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

# Redondeamos la temperatura para agrupar mejor
df['TMP_round'] = df['TMP'].round(0)

# Lista de contaminantes a analizar
contaminantes = ['PM2.5','PM10','NOx','O3','CO','NO','NO2','HR','PB','PP','RS']

# Crear subcarpetas por tipo de gráfico
os.makedirs("./plots/timelines", exist_ok=True)
os.makedirs("./plots/heatmap", exist_ok=True)
os.makedirs("./plots/bar", exist_ok=True)

# ------------------ 1. Timeline anual de cada contaminante ------------------
# Tipo: Line plot
# Para que sirve: Ver evolución anual de cada contaminante
# Objetivo del plot: Identificar tendencias anuales y comparar contaminantes
for c in contaminantes:
    plt.figure(figsize=(12,6))
    timeline = df.groupby('year')[c].mean()
    plt.plot(timeline.index, timeline.values, marker='o')
    plt.title(f"Evolución anual de {c}")
    plt.xlabel("Año")
    plt.ylabel(f"{c} promedio")
    plt.tight_layout()
    plt.savefig(f"./plots/timelines/timeline_{c}.png")
    plt.close()

# ------------------ 2. Heatmap de todos los contaminantes vs mes ------------------
# Tipo: Heatmap
# Para que sirve: Promedio mensual de cada contaminante
# Objetivo del plot: Detectar meses con mayor contaminación
contam_mes = df.groupby('month')[contaminantes].mean()
plt.figure(figsize=(10,6))
sns.heatmap(contam_mes.T, cmap="Reds", annot=True, fmt=".2f", cbar_kws={'label':'Concentración promedio'})
plt.title("Promedio mensual de contaminantes")
plt.xlabel("Mes")
plt.ylabel("Contaminante")
plt.tight_layout()
plt.savefig("./plots/heatmap/contaminantes_vs_mes.png")
plt.close()

# ------------------ 3. Line plot mes vs temperatura por año ------------------
# Tipo: Line plot multianual
# Para que sirve: Ver evolución de la temperatura media por mes a lo largo de los años
# Objetivo del plot: Identificar tendencias estacionales y cambios interanuales
temp_month_year = df.groupby(['year','month'])['TMP_round'].mean().unstack(level=0)
plt.figure(figsize=(12,6))
for year in temp_month_year.columns:
    plt.plot(temp_month_year.index, temp_month_year[year], marker='o', label=str(year))
plt.title("Temperatura promedio mensual por año (todas las estaciones)")
plt.xlabel("Mes")
plt.ylabel("Temperatura promedio (°C)")
plt.legend(title="Año", bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig("./plots/timelines/temp_vs_month_year.png")
plt.close()

# ------------------ 3b. Line plot mes vs temperatura por estación ------------------
# Tipo: Line plot multianual por estación
# Para que sirve: Ver cómo la temperatura promedio mensual ha cambiado por año en cada estación
# Objetivo del plot: Analizar patrones locales de temperatura en estaciones específicas
stations = df['station_name'].unique()
for station in stations:
    df_station = df[df['station_name'] == station]
    temp_month_year_station = df_station.groupby(['year','month'])['TMP_round'].mean().unstack(level=0)
    plt.figure(figsize=(12,6))
    for year in temp_month_year_station.columns:
        plt.plot(temp_month_year_station.index, temp_month_year_station[year], marker='o', label=str(year))
    plt.title(f"Temperatura promedio mensual por año - Estación: {station}")
    plt.xlabel("Mes")
    plt.ylabel("Temperatura promedio (°C)")
    plt.legend(title="Año", bbox_to_anchor=(1.05,1))
    safe_name = station.replace(" ", "_").replace("/","_")
    plt.tight_layout()
    plt.savefig(f"./plots/timelines/temp_vs_month_year_{safe_name}.png")
    plt.close()

# ------------------ 4. Gráfico de barras: promedio PM2.5 por estacion ------------------
# Tipo: Bar plot
# Para que sirve: Comparar PM2.5 promedio entre estaciones
# Objetivo del plot: Identificar estaciones con mayor contaminación crónica de PM2.5
pm25_estacion_avg = df.groupby('station_name')['PM2.5'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=pm25_estacion_avg.index, y=pm25_estacion_avg.values)
plt.title("Promedio de PM2.5 por estacion")
plt.xlabel("estacion")
plt.ylabel("PM2.5 promedio (µg/m³)")
plt.tight_layout()
plt.savefig("./plots/bar/PM2.5_por_estacion.png")
plt.close()

# ------------------ 5. Heatmap de correlación entre todos los contaminantes ------------------
# Tipo: Heatmap
# Para que sirve: Visualizar correlación entre contaminantes
# Objetivo del plot: Identificar si contaminantes se comportan de manera similar o si comparten fuentes
plt.figure(figsize=(10,8))
corr_matrix = df[contaminantes].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label':'Correlación'})
plt.title("Matriz de correlación entre contaminantes")
plt.tight_layout()
plt.savefig("./plots/heatmap/corr_contaminantes.png")
plt.close()

# ------------------ 6. Heatmap mes vs temperatura (registros) ------------------
# Tipo: Heatmap
# Para que sirve: Ver distribución de registros por mes y temperatura
# Objetivo del plot: Permite observar meses y rangos de temperatura más frecuentes
heat_month_temp = df.groupby(['month','TMP_round']).size().unstack(fill_value=0)
plt.figure(figsize=(12,6))
sns.heatmap(heat_month_temp, cmap="YlGnBu", cbar_kws={'label':'Cantidad de registros'})
plt.title("Distribución de registros por mes y temperatura")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Mes")
plt.tight_layout()
plt.savefig("./plots/heatmap/mes_vs_TMP_count.png")
plt.close()

# ------------------ 7. Line plot: contaminantes por mes y año ------------------
# Tipo: Line plot multianual mensual
# Para que sirve: Ver cómo cada contaminante varía a lo largo de los meses para cada año
# Objetivo del plot: Permite identificar patrones estacionales y cambios interanuales de los contaminantes
for c in contaminantes:
    plt.figure(figsize=(12,6))
    month_year = df.groupby(['year','month'])[c].mean().unstack(level=0)
    for year in month_year.columns:
        plt.plot(month_year.index, month_year[year], marker='o', label=str(year))
    plt.title(f"Evolución mensual de {c} por año")
    plt.xlabel("Mes")
    plt.ylabel(f"{c} promedio (µg/m³)")
    plt.legend(title="Año", bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(f"./plots/timelines/{c}_month_year.png")
    plt.close()

# ------------------ 8. Bar plot: temperatura máxima por estacion ------------------
# Tipo: Bar plot
# Para que sirve: Comparar la temperatura máxima registrada en cada estación/estacion
# Objetivo del plot: Permite identificar cuáles estaciones han alcanzado las temperaturas más altas históricamente
max_temp_mun = df.groupby('station_name')['TMP'].max().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=max_temp_mun.index, y=max_temp_mun.values)
plt.title("Temperatura máxima histórica por estacion")
plt.xlabel("Estación")
plt.ylabel("Temperatura máxima (°C)")
plt.xticks(rotation=45, ha='right')  # Rota los nombres de las estaciones para mejor lectura
plt.tight_layout()
plt.savefig("./plots/bar/max_temp_estacion.png")
plt.close()



# Despues de la generacion de todos esos graficos podemos identificar los que nos empiezan a dar datos interesantes acerca de las mediciones.
# Por ejemplo, en las graficas de barras pudimos ver que en promedio, Juarez es el que tiene en promedio mayor concentracion de particulas PM2.5
# En las graficas de heatmap, pudimos ver que no existe correlacion entre los contaminantes
# Descubrimos que al parecer, en su mayoria los registros son de Agosto, con una temperatura de 29°C
# Tambien en los timelines pudimos notar que justo al inicio de pandemia en Marzo de 2020 se dio el pico mas alto de particulas PM2.5 entre los años 2016 y 2021
# Justo en la misma pandemia, pudimos ver como el Monoxido de Carbono (CO) fue a la alza hasta Diciembre de 2020, afortunadamente empezo a disminuir considerablemente durante el 2021.
# En cuanto a la evolucion anual de particulas PM2.5, vimos que desde 2017 fueron a la alza, hasta que en 2020 hubo una caida importante desde ese año hasta el 2021
# La misma evolucion anual pero ahora del Ozono troposférico (O3) nos hizo notar que desde 2018 fue a la alza.
# Por ultimo, pudimos notar que entre esos años, la temperatura mas alta se registro en la estacion de La Pastora.
# Como observacion, Cadereyta es la estacion que tiene mejores registros, es decir la estacion que ha tenido los datos mas completos sin faltantes aun despues de la limpieza.