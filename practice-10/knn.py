# Semana 10: Data classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Creamos la carpeta para guardar resultados
os.makedirs("./plots/knn", exist_ok=True)

# Cargamos el dataset
df = pd.read_csv("../dataset/monterrey_aq.csv")

features = ['PM10', 'NOx', 'O3', 'CO', 'NO', 'NO2', 'HR', 'TMP', 'PB', 'PP', 'RS']
target = 'PM2.5'

# Clasificación basada en PM2.5 según criterios de la OMS
bins = [0, 12, 35.4, 55.4, np.inf]
labels = ["Bajo", "Moderado", "Alto", "Muy alto"]

df["PM2.5_level"] = pd.cut(df["PM2.5"], bins=bins, labels=labels)
df = df.dropna(subset=["PM2.5_level"])

# Separamos las variables predictoras y objetivo
X = df[features]
y = df["PM2.5_level"]

# Normalizamos las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividimos los datos en conjunto de entrenamiento y prueba (75% - 25%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Entrenamos el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluamos el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Creamos una matriz de confusión para visualizar los resultados de la clasificación
# Sirve para ver en qué categorías el modelo se está equivocando
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title("Matriz de confusión - KNN (Clasificación PM2.5)")
plt.tight_layout()
plt.savefig("./plots/knn/confusion_matrix.png")
plt.close()

# Optimizamos el valor de K probando diferentes valores y viendo su impacto en la precisión
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker='o')
plt.title("Precisión del modelo según número de vecinos (k)")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/knn/k_optimization.png")
plt.close()

# Conclusion:

# El modelo KNN muestra una precisión razonable de 75.9% para clasificar los niveles de PM2.5.
# identificando correctamente principalmente los niveles “Moderado” y “Bajo” de PM2.5.

# Sin embargo, las clases “Alto” y “Muy alto” presentan menor desempeño debido a la escasez de muestras, reflejado en el bajo recall y F1-score. La matriz de confusión evidencia que los errores ocurren sobre todo en estas categorías extremas. 

# La optimización de K muestra que K=5 ofrece un buen equilibrio entre estabilidad y precisión. KNN es útil para predecir niveles comunes de PM2.5, pero requiere más datos o variables adicionales para mejorar la detección de episodios críticos.
