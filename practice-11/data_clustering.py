# Semana 11: Data Clustering

# Importamos las librerías necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Creamos la carpeta para guardar resultados
os.makedirs("./plots/", exist_ok=True)

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Seleccionamos las variables relevantes
features = ['PM10', 'NOx', 'O3', 'CO', 'NO', 'NO2', 'HR', 'TMP', 'PB', 'PP', 'RS', 'PM2.5']
df = df.dropna(subset=features)

# Estandarizamos las características para que todas tengan la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Utilizamos el método del codo para determinar el número óptimo de clusters
inertia = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title("Método del codo - Selección del número óptimo de clusters (K)")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia (Suma de distancias cuadradas)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/elbow_method.png")
plt.close()

# Basados en nuestra observación del gráfico, seleccionamos K=3 como el número óptimo de clusters ya que tiene la mejor relación entre inercia y número de clusters (7000 puntos aproximadamente entre 2 y 3).
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Calculamos el puntaje de Silhouette para evaluar la calidad del clustering
sil_score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {sil_score:.3f}")

# Ahora, visualizamos los resultados del clustering
# Distribución de observaciones por cluster
plt.figure(figsize=(8, 5))
sns.countplot(x="Cluster", data=df)
plt.title("Distribución de observaciones por cluster")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de observaciones")
plt.tight_layout()
plt.savefig("./plots/cluster_distribution.png")
plt.close()

# Relación entre PM2.5 y temperatura
plt.figure(figsize=(7, 6))
sns.scatterplot(x=df["TMP"], y=df["PM2.5"], hue=df["Cluster"], alpha=0.7)
plt.title("Clusters en función de la temperatura y PM2.5")
plt.xlabel("Temperatura (TMP)")
plt.ylabel("PM2.5")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("./plots/pm25_vs_tmp_clusters.png")
plt.close()

# Relación entre humedad y PM2.5
plt.figure(figsize=(7, 6))
sns.scatterplot(x=df["HR"], y=df["PM2.5"], hue=df["Cluster"], alpha=0.7)
plt.title("Clusters en función de la humedad y PM2.5")
plt.xlabel("Humedad relativa (HR)")
plt.ylabel("PM2.5")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("./plots/pm25_vs_hr_clusters.png")
plt.close()

# Resumen de las características promedio por cluster
cluster_summary = df.groupby("Cluster")[features].mean()
print("\nResumen promedio por cluster:")
print(cluster_summary)

"""
CONCLUSIÓN:
El modelo permitió agrupar los datos ambientales de Monterrey en tres patrones principales
de comportamiento atmosférico. Cada clúster representa un tipo de condición ambiental distinta:

El primer grupo (Cluster 1) corresponde a días con buena calidad del aire, caracterizados por bajas
concentraciones de contaminantes. Representa los días de invierno u otoño limpios. La explicación más probable es que estos son días con viento o que acaban de ser "barridos" por un frente frío. El viento dispersa activamente toda la contaminación de la cuenca, resultando en un aire limpio a pesar de las temperaturas frescas.

El segundo grupo (Cluster 0) agrupa días con contaminación intermedia, temperaturas más altas y mayor
radiación solar, lo que puede favorecer la dispersión de partículas. Representa los días de verano o primavera. El intenso calor solar genera convección (el aire caliente sube), lo que ayuda a dispersar los contaminantes y a mantener la calidad del aire en niveles aceptables, aunque no sea perfectamente limpia.

El tercer grupo (Cluster 2) refleja días con alta contaminación, concentraciones elevadas de PM2.5,
PM10, NOx y CO, ocurre en temperaturas moderadas (10-25°C) y con alta humedad. Está casi exclusivamente asociado con niveles de humedad de moderados a altos. Representa el peor escenario, un día templado, húmedo y sin viento. La humedad y la falta de dispersión (probablemente por una inversión térmica) "atrapan" los contaminantes y facilitan reacciones químicas que crean más PM2.5. Este es el día en que se emiten las alertas ambientales.

Aunque el puntaje de Silhouette (0.163) indica una separación moderada entre grupos, los resultados
permiten identificar patrones relevantes y establecer una base para un análisis estacional o predictivo
sobre la calidad del aire en Monterrey (Nuevo Leon en general).

Como imaginamos, es un peligro en Monterrey cuando estamos dentro del Cluster 2 debido a la alta contaminación, esto es cuando vemos los cerros con smog y sucede porque la alta humedad hace que la luz se disperse en las partículas de contaminación, haciéndolas mucho más visibles. Lo que vemos no es solo neblina si no miles de partículas con humedad.
"""
