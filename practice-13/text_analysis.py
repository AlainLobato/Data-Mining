# Practica 13: Text Analysis

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Creamos la carpeta para guardar los plots
os.makedirs("plots", exist_ok=True)

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Concatenamos todos los nombres de las estaciones en un solo texto
text_data = " ".join(df["station_name"].astype(str))

# Generamos la nube de palabras
wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color="white",
    colormap="plasma",
    max_words=100
).generate(text_data)

# Creamos el plot de la nube de palabras y lo guardamos
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de Palabras: Frecuencia de Estaciones de Monitoreo", fontsize=14)
plt.tight_layout()
plt.savefig("plots/wordcloud.png", dpi=300)
plt.close()

# Conclusión
"""
Este análisis de texto utiliza los nombres de las estaciones de monitoreo (station_name) presentes en el dataset de
calidad del aire de Monterrey.

A través de una nube de palabras se puede observar visualmente cuáles estaciones tienen mayor
frecuencia de registro en los datos.

Las estaciones que aparecen con un tamaño mayor, como Cadereyta, indican una mayor cantidad de observaciones, lo que
puede relacionarse con su actividad, cobertura geográfica o disponibilidad de datos. En este caso, Cadereyta, Obispado y
Santa Catarina son algunas de las estaciones más prominentes, sugiriendo que estas áreas tienen una mayor densidad de
monitoreo o actividad relacionada con la calidad del aire. Por otra parte, estaciones con nombres más pequeños indican
una menor frecuencia de datos, como lo es Juárez o Apodaca.

Esta técnica de análisis textual facilita la identificación rápida de patrones en datos categóricos, y complementa los
análisis numéricos previos realizados con PM2.5 y AQI.
"""
