# PIA: Análisis Completo del Dataset de Calidad del Aire

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Creamos la carpeta de plots si no existe
os.makedirs("plots", exist_ok=True)

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Convertimos datetime
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Eliminamos NaN solo en columnas críticas
df = df.dropna(subset=["datetime", "PM2.5", "TMP"])

# Extraemos año y mes
df["YEAR"] = df["datetime"].dt.year
df["MONTH"] = df["datetime"].dt.month

# Análisis Exploratorio Inicial

# Correlación
plt.figure(figsize=(10, 7))
sns.heatmap(df[["PM2.5", "PM10", "NOx", "O3", "CO", "TMP"]].corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.savefig("plots/correlation_matrix.png")
plt.close()

# PM2.5 Mensual

# Agrupamos por año y mes
pm25_monthly = df.groupby(["YEAR", "MONTH"])["PM2.5"].mean().reset_index()

# Gráfica de línea
plt.figure(figsize=(12, 6))
plt.plot(pm25_monthly.index, pm25_monthly["PM2.5"])
plt.title("Tendencia Mensual de PM2.5 (2016–2021)")
plt.xlabel("Tiempo (Índice mensual)")
plt.ylabel("PM2.5")
plt.savefig("plots/pm25_monthly_trend.png")
plt.close()

# Heatmap año × mes
pm25_pivot = pm25_monthly.pivot(index="YEAR", columns="MONTH", values="PM2.5")
plt.figure(figsize=(10, 6))
sns.heatmap(pm25_pivot, cmap="YlOrRd", annot=True, fmt=".1f")
plt.title("PM2.5 Promedio por Año y Mes")
plt.savefig("plots/pm25_heatmap.png")
plt.close()

# AQI Anual

# Fórmula simplificada de AQI para PM2.5
def calculate_aqi(pm25):
    return (50/12) * pm25  # escala aproximada

df["AQI"] = df["PM2.5"].apply(calculate_aqi)

aqi_yearly = df.groupby("YEAR")["AQI"].mean()

plt.figure(figsize=(10, 6))
aqi_yearly.plot(kind="bar")
plt.title("Promedio Anual de AQI (2016–2021)")
plt.ylabel("AQI")
plt.savefig("plots/aqi_yearly.png")
plt.close()

# MODELOS DE FORECASTING PARA PM2.5

# Serie temporal mensual
ts = pm25_monthly["PM2.5"]

# Modelo Lineal
X = np.array(pm25_monthly.index).reshape(-1, 1)
y = ts.values

model_lin = LinearRegression()
model_lin.fit(X, y)

# Predicción 12 meses
future_index = np.arange(len(X), len(X) + 12).reshape(-1, 1)
pred_lin = model_lin.predict(future_index)

# Guardar gráfica
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label="Histórico")
plt.plot(future_index, pred_lin, label="Predicción Lineal")
plt.title("Predicción Lineal de PM2.5 para 2022")
plt.legend()
plt.savefig("plots/forecast_linear_pm25.png")
plt.close()

print("=== MODELO LINEAL PM2.5 ===")
print("MSE:", mean_squared_error(y, model_lin.predict(X)))
print("R²:", r2_score(y, model_lin.predict(X)))
print("Predicción promedio 2022 (Lineal):", pred_lin.mean(), "µg/m³\n")

# Modelo ARIMA
model_arima = ARIMA(ts, order=(1,1,1)).fit()
pred_arima = model_arima.forecast(12)

plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label="Histórico")
plt.plot(pred_arima.index, pred_arima.values, label="Predicción ARIMA")
plt.title("Predicción ARIMA de PM2.5 para 2022")
plt.legend()
plt.savefig("plots/forecast_arima_pm25.png")
plt.close()

print("=== MODELO ARIMA (PM2.5) ===")
print(pred_arima)
print("Promedio PM2.5 2022 (ARIMA):", float(pred_arima.mean()), "µg/m³\n")

# Modelo SARIMA
model_sarima = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
pred_sarima = model_sarima.forecast(12)

plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label="Histórico")
plt.plot(pred_sarima.index, pred_sarima.values, label="Predicción SARIMA")
plt.title("Predicción SARIMA de PM2.5 para 2022")
plt.legend()
plt.savefig("plots/forecast_sarima_pm25.png")
plt.close()

print("=== MODELO SARIMA (PM2.5) ===")
print(pred_sarima)
print("Promedio PM2.5 2022 (SARIMA):", float(pred_sarima.mean()), "µg/m³\n")

# Wordcloud de station_name

text = " ".join(df["station_name"].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("plots/wordcloud_station_name.png")
plt.close()
