import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Creamos la carpeta para guardar resultados
os.makedirs("plots", exist_ok=True)

# Conversión de fechas
df["datetime"] = pd.to_datetime(df["datetime"])

# Creamos variables de año y mes
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month

# Agrupamos por mes y año para obtener promedios mensuales de PM2.5
monthly = df.groupby(["year", "month"], as_index=False)["PM2.5"].mean()
monthly["datetime"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-15")

# Definimos una función para convertir PM2.5 a AQI según la tabla de la EPA
def pm25_to_aqi(pm25):
    if pm25 <= 12:
        return (50/12)*pm25
    elif pm25 <= 35.4:
        return 50 + (100-50)/(35.4-12)*(pm25-12)
    elif pm25 <= 55.4:
        return 100 + (150-100)/(55.4-35.4)*(pm25-35.4)
    elif pm25 <= 150.4:
        return 150 + (200-150)/(150.4-55.4)*(pm25-55.4)
    elif pm25 <= 250.4:
        return 200 + (300-200)/(250.4-150.4)*(pm25-150.4)
    elif pm25 <= 350.4:
        return 300 + (400-300)/(350.4-250.4)*(pm25-250.4)
    else:
        return 400 + (500-400)/(500-350.4)*(pm25-350.4)

monthly["AQI"] = monthly["PM2.5"].apply(pm25_to_aqi)

# Entrenamos modelos de predicción
# Modelo Lineal
monthly["time_index"] = (monthly["year"] - monthly["year"].min()) * 12 + monthly["month"]
X = monthly[["time_index"]]
y_pm = monthly["PM2.5"]
y_aqi = monthly["AQI"]

model_pm = LinearRegression().fit(X, y_pm)
model_aqi = LinearRegression().fit(X, y_aqi)

monthly["PM2.5_pred_lin"] = model_pm.predict(X)
monthly["AQI_pred_lin"] = model_aqi.predict(X)

# Predicción 2022 (Lineal)
last_index = monthly["time_index"].max()
future_months = np.arange(last_index + 1, last_index + 13).reshape(-1, 1)

pred_pm2022_lin = model_pm.predict(future_months)
pred_aqi2022_lin = model_aqi.predict(future_months)

# Modelo ARIMA
# Convertimos el promedio mensual en serie temporal
ts_pm = monthly.set_index("datetime")["PM2.5"]

# Ajustamos un modelo ARIMA simple
arima_pm = ARIMA(ts_pm, order=(2, 1, 2))
model_arima_pm = arima_pm.fit()

# Predicción mensual de 12 pasos hacia adelante (2022)
forecast_arima_pm = model_arima_pm.forecast(steps=12)
forecast_dates = pd.date_range(ts_pm.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")

# Modelo SARIMA
ts_pm = monthly.set_index("datetime")["PM2.5"]

# Ajustamos un modelo SARIMA simple
sarima_model = SARIMAX(ts_pm, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_results = sarima_model.fit(disp=False)

# Predicción 12 meses adelante
forecast_sarima_pm = sarima_results.get_forecast(steps=12).predicted_mean
forecast_dates = pd.date_range(ts_pm.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")

# Graficamos los resultados de las predicciones
# PM2.5 SARIMA vs Real
plt.figure(figsize=(10, 5))
plt.plot(monthly["datetime"], monthly["PM2.5"], label="Real PM2.5", marker='o')
plt.plot(monthly["datetime"], monthly["PM2.5_pred_lin"], label="Predicción Lineal", linestyle="--")
plt.plot(forecast_dates, forecast_sarima_pm, label="Predicción SARIMA 2022", color="red", linestyle=":")
plt.title("Predicción Mensual de PM2.5 (Lineal vs SARIMA)")
plt.xlabel("Fecha")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/forecast_pm25_sarima.png", dpi=300)
plt.close()

print("\n=== MODELO SARIMA (PM2.5) ===")
print(forecast_sarima_pm)
print(f"\nPromedio PM2.5 2022 (SARIMA): {forecast_sarima_pm.mean():.2f} µg/m³")

# PM2.5 Lineal vs Real
plt.figure(figsize=(10, 5))
plt.plot(monthly["datetime"], monthly["PM2.5"], label="Real PM2.5", marker='o')
plt.plot(monthly["datetime"], monthly["PM2.5_pred_lin"], label="Predicción Lineal", linestyle="--")
plt.plot(forecast_dates, forecast_arima_pm, label="Predicción ARIMA 2022", color="red", linestyle=":")
plt.title("Predicción Mensual de PM2.5 (Lineal vs ARIMA)")
plt.xlabel("Fecha")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/forecast_pm25_arima.png", dpi=300)
plt.close()

# AQI Lineal
plt.figure(figsize=(10, 5))
plt.plot(monthly["datetime"], monthly["AQI"], label="AQI Real", marker='o')
plt.plot(monthly["datetime"], monthly["AQI_pred_lin"], label="Predicción Lineal", linestyle="--")
plt.title("Predicción Mensual del Índice de Calidad del Aire (Lineal)")
plt.xlabel("Fecha")
plt.ylabel("AQI (Índice EPA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/forecast_aqi_lineal.png", dpi=300)
plt.close()

# Evaluamos los modelos
mse_pm = mean_squared_error(y_pm, monthly["PM2.5_pred_lin"])
r2_pm = r2_score(y_pm, monthly["PM2.5_pred_lin"])
mse_aqi = mean_squared_error(y_aqi, monthly["AQI_pred_lin"])
r2_aqi = r2_score(y_aqi, monthly["AQI_pred_lin"])

# Resultados
print("\n=== MODELO LINEAL PM2.5 ===")
print(f"MSE: {mse_pm:.3f}")
print(f"R²: {r2_pm:.3f}")
print(f"Predicción promedio 2022 (Lineal): {pred_pm2022_lin.mean():.2f} µg/m³")

print("\n=== MODELO LINEAL AQI ===")
print(f"MSE: {mse_aqi:.3f}")
print(f"R²: {r2_aqi:.3f}")
print(f"Predicción promedio 2022 (Lineal): {pred_aqi2022_lin.mean():.2f}")

print("\n=== MODELO ARIMA (PM2.5) ===")
print(forecast_arima_pm)
print(f"\nPromedio PM2.5 2022 (ARIMA): {forecast_arima_pm.mean():.2f} µg/m³")

# Conclusión del análisis de modelos
"""
El análisis de predicción de PM2.5 y AQI se realiza utilizando tres enfoques: un modelo Lineal,
un modelo ARIMA y un modelo SARIMA. Cada uno ofrece una perspectiva distinta sobre el comportamiento
temporal de la contaminación en Monterrey, permitiendo comprender tanto la tendencia general como
las posibles fluctuaciones estacionales.

Modelo Lineal:
Este modelo proporciona una visión base del comportamiento de los contaminantes a lo largo del tiempo.
Su coeficiente de determinación (R²) es bajo, lo que indica que la relación entre los valores de PM2.5
y el paso de los años no sigue una tendencia estrictamente lineal. Sin embargo, su predicción promedio
de 21.11 µg/m³ refleja una estabilidad relativa en las concentraciones de partículas, sugiriendo que los
niveles de contaminación se mantienen en un rango constante.

Modelo ARIMA:
El modelo ARIMA incorpora la dependencia temporal entre observaciones pasadas, permitiendo capturar la
inercia del sistema atmosférico. Su predicción promedio de 19.62 µg/m³ muestra un comportamiento suave
y estable, lo que indica que las variaciones mensuales no alteran significativamente la tendencia global.
Este modelo representa una aproximación intermedia entre la simplicidad del modelo lineal y la complejidad
estacional del SARIMA.

Modelo SARIMA:
Este modelo amplía las capacidades del ARIMA al incluir componentes estacionales, logrando reflejar mejor
los ciclos de aumento y disminución de las partículas finas a lo largo del año. Las predicciones mensuales
varían entre 16 y 24 µg/m³, con un promedio de 20.09 µg/m³. Esto evidencia que existe una dinámica
estacional en la contaminación, posiblemente asociada a factores como la temperatura, la humedad y la
actividad humana en diferentes épocas del año.

Los tres modelos ofrecen una visión complementaria del fenómeno, el modelo Lineal permite
identificar la estabilidad general, ARIMA describe la persistencia temporal del sistema y SARIMA revela
patrones estacionales más complejos. Los resultados muestran que las concentraciones de PM2.5 tienden a
mantenerse dentro de un rango constante en los últimos años, sin incrementos abruptos, lo que sugiere una
relativa estabilidad ambiental, aunque con variaciones cíclicas que conviene seguir monitoreando.
"""
