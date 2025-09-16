# Practica 5: Estadísticas Descriptivas

# Cargamos nuestra libreria de pandas
import pandas as pd

# Cargamos nuestro dataset ya limpio
df = pd.read_csv('../dataset/monterrey_aq.csv')

# Vamos a obtener estadisticas numericas
print(df.describe())

# Ahora estadísticas para variables categóricas
print(df.describe(include=['object']))

# Crear columna de fecha
df['date'] = pd.to_datetime(df['datetime']).dt.date

# Agrupar por fecha
grouped_date = df.groupby('date').agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'O3': 'mean',
    'CO': 'mean',
    'NO2': 'mean',
    'NOx': 'mean'
}).reset_index()

print(grouped_date.head())