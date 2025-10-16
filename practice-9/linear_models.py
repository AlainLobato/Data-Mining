# Semana 9 - Linear Models + Correlation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid")
os.makedirs("./plots/linear", exist_ok=True)
os.makedirs("./plots/random_forest", exist_ok=True)
os.makedirs("./plots/correlation", exist_ok=True)

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

# Apartamos las variables predictoras y objetivo
features = ["PM10", "NOx", "O3", "CO", "NO", "NO2", "HR", "TMP", "PB", "PP", "RS"]
target = "PM2.5"

# Separamos los datos en X e Y (variables independientes y dependiente)
X = df[features]
y = df[target]

# Dividimos en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un modelo de regresión lineal con nuestro conjunto de entrenamiento
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluamos el modelo con R²
r2_linear = r2_score(y_test, y_pred_linear)
print(f"R² del modelo Lineal: {r2_linear:.4f}")

# Calculamos los coeficientes del modelo
coef = pd.Series(linear_model.coef_, index=features).sort_values()
print("Coeficientes del modelo Lineal:")
print(coef)

# Generamos gráficos para visualizar los resultados
plt.figure(figsize=(10,6))
coef.plot(kind="barh", color="teal")
plt.title("Importancia de las variables - Regresión Lineal")
plt.xlabel("Peso del coeficiente")
plt.tight_layout()
plt.savefig("./plots/linear/coef_importance.png")
plt.close()

# Generamos un comparativo entre valores reales y predichos
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f"Regresión Lineal: Real vs Predicho (R²={r2_linear:.3f})")
plt.xlabel("PM2.5 Real")
plt.ylabel("PM2.5 Predicho")
plt.tight_layout()
plt.savefig("./plots/linear/real_vs_pred.png")
plt.close()


# Para buscar un modelo más robusto, probamos con Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Entrenamos el modelo
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"R² del modelo Random Forest: {r2_rf:.4f}")
print(f"MSE del modelo Random Forest: {mse_rf:.4f}")

# Calculamos la importancia de las variables en Random Forest
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Importancia de variables - Random Forest:")
print(importances)

# Generamos gráficos para visualizar los resultados de Random Forest
plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Importancia de variables - Random Forest")
plt.xlabel("Importancia relativa")
plt.tight_layout()
plt.savefig("./plots/random_forest/variable_importance.png")
plt.close()

# Gráfico comparativo entre valores reales y predichos por Random Forest
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title(f"Random Forest: PM2.5 Real vs Predicho (R²={r2_rf:.3f})")
plt.xlabel("Valor real PM2.5")
plt.ylabel("Valor predicho PM2.5")
plt.tight_layout()
plt.savefig("./plots/random_forest/real_vs_pred.png")
plt.close()

# Analizamos la correlación entre las variables
plt.figure(figsize=(10,8))
sns.heatmap(df[[target] + features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación entre contaminantes y meteorología")
plt.tight_layout()
plt.savefig("./plots/correlation/heatmap_correlation.png")
plt.close()

# Gráficos de dispersión entre PM2.5 y otras variables
os.makedirs("./plots/scatter_vars", exist_ok=True)
for var in ["PM10", "NOx", "NO2", "O3", "CO", "PP"]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[var], y=df["PM2.5"], alpha=0.4)
    sns.regplot(x=df[var], y=df["PM2.5"], scatter=False, color="red")
    plt.title(f"Relación entre {var} y PM2.5")
    plt.xlabel(var)
    plt.ylabel("PM2.5")
    plt.tight_layout()
    plt.savefig(f"./plots/scatter_vars/{var}_vs_PM2.5.png")
    plt.close()




# Despues de crear los dos modelos, podemos concluir que el modelo de Random Forest es más robusto y preciso para predecir los niveles de PM2.5 en Monterrey, ya que maneja mejor la complejidad y no asume linealidad entre las variables. Además, la importancia de las variables nos ayuda a entender qué factores influyen más en la contaminación por PM2.5.

# R2 en modelo lineal: 0.2098 es un valor bajo, indicando que el modelo lineal no captura bien la variabilidad de los datos.
# Hay alguna correlación lineal, pero la mayor parte del comportamiento del PM2.5 no es lineal. Los contaminantes y condiciones meteorológicas interactúan de formas más complejas que un modelo lineal no puede capturar.

# Coeficientes del modelo Lineal: 
# NO -42.962653
# PP -0.921790
# PB -0.010492
# RS 0.001457
# TMP 0.113177
# HR 0.115391
# PM10 0.159845
# CO 0.980632
# NO2 25.358782
# O3 48.429076
# NOx 79.150645

# El modelo sugiere que las emisiones (NOx, O₃, NO₂) son los principales factores que elevan PM2.5, mientras que la lluvia (PP) y el NO lo reducen.

# R2 en modelo Random Forest: 0.35 es mejor que el lineal, pero aún indica que hay mucha variabilidad no explicada. El MSE de 102.28 indica que en promedio, las predicciones del modelo se desvían en aproximadamente 10.11 unidades de PM2.5. Refuerza que Random Forest es más preciso para predecir PM2.5.

# Importancia de variables - Random Forest:
# PM10 0.387879
# HR 0.114247
# CO 0.081031
# NO2 0.071193
# TMP 0.070445
# PB 0.067314
# O3 0.060588
# RS 0.048618
# NO 0.046607
# NOx 0.046047
# PP 0.006031

# PM10 es la variable dominante. Factores meteorológicos (HR, TMP, PB) también son importantes, lo que el modelo lineal no captaba bien. Random Forest combina relaciones no lineales, por eso obtiene mejor desempeño.

# Conclusión:

# Limitación real: La regresión lineal simple no refleja la física de la atmósfera, especialmente los procesos fotoquímicos que generan PM2.5.

# La concentración de PM2.5 depende principalmente de:

# PM10, como indicador de material particulado total. Partículas gruesas se descomponen o se mezclan y contribuyen a PM2.5.

# Humedad y temperatura, que influyen en la dispersión y formación química. Alta humedad puede aumentar la concentración de partículas suspendidas.

# CO, NO₂ y NOx, que reflejan fuentes de combustión (vehículos, fábricas). Se transforman químicamente en partículas secundarias.

# Presión y radiación solar, que afectan la estabilidad del aire y la fotodisociación.

# Precipitación, que limpia el aire, disminuyendo PM2.5. Reduce PM2.5 por lavado atmosférico.

# El modelo Random Forest representa mejor la realidad atmosférica, porque las relaciones entre gases y partículas no son lineales.

# La importancia de PM10, gases contaminantes y condiciones meteorológicas es consistente con estudios de calidad del aire en ciudades.
