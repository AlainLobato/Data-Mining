# Practica 4: Data Cleaning

# Cargamos nuestra libreria de pandas
import pandas as pd

# Cargamos nuestros datasets
df1 = pd.read_csv('../datasets/stations_daily.csv')
df2 = pd.read_csv('../datasets/stations_rsinaica.csv')

# Vemos las primeras filas de nuestro dataset
print("-"*40)
print("Dataset 1:")
print(df1.head())

print("-"*40)
print("Dataset 2:")
print(df2.head())

# Buscamos en la columna network_name si existe el valor de Monterrey, si existe guardamos los stations_id
print("-"*40)
print("Network Name:")
stations_id = []

for code in df2['network_name'].unique():
    if not pd.isna(code) and code.startswith('Monterrey'):
        stations_id.extend(df2[df2['network_name'] == code]['station_id'].tolist())
        print(df2[df2['network_name'] == code])
        
print("Stations ID:", stations_id)

# Juntamos ambos datasets a partir de la columna station_id
df = pd.merge(df1, df2, on='station_id')
print("Dataframe merged shape:", df.shape)

# Filtramos el dataset para quedarnos solo con las estaciones de Monterrey e imprimimos su tamaño
df_mty = df[df['station_id'].isin(stations_id)]
print("Dataframe Monterrey shape:", df_mty.shape)

# Podemos notar una disminucion muy grande entre los datos, bajando de 231,592 registros a 41,403 registros

# Ahora vamos a obtener las columnas que tienen datos nulos y cuantos datos nulos tienen para saber si es necesario limpiarlos
print("-"*40)
print("Missing values per column:")
print(df_mty.isnull().sum().sort_values(ascending=False))

# Ahora podemos ver columnas que tienen 80% o mas de sus datos nulos y columnas que no necesitamos, por lo que las eliminamos

cols_to_drop = [
    # completamente vacías
    'HCNM', 'HCT', 'HRI', 'CH4', 'BEN', 'IUV', 'H2S', 'CO2', 'CN',
    'UVA', 'XIL', 'TMPI', 'PST', 'date_validated', 'video_interior', 'interior',
    
    # metadatos administrativos o irrelevantes
    'video', 'street_view', 'color', 'ext', 'colonia', 'street', 'zip',
    'address', 'passed_validation', 'date_validated2', 'year_started', 'date_started',
    
    # identificadores redundantes
    'network_id', 'network_name', 'network_code', 'station_code'
]

df_mty_cleaned = df_mty.drop(columns=cols_to_drop)
print("Dataframe cleaned shape:", df_mty_cleaned.shape)

# Para las filas que tengan valores nulos las eliminaremos para evitar asi problemas en futuros analisis
df_airq = df_mty_cleaned.dropna(subset=['NOx', 'NO', 'HR', 'TMP', 'PB', 'PP', 'RS', 'NO2', 'PM2.5', 'O3', 'CO', 'PM10'])
print("Dataframe after handling missing values shape:", df_airq.shape)

# Finalmente guardamos el dataset limpio en un nuevo archivo csv
df_airq.to_csv('../dataset/monterrey_aq.csv', index=False)
print("Cleaned dataset saved to '../dataset/monterrey_aq.csv'")

# Vemos las primeras filas de nuestro dataset limpio, comprobando tambien que no existen valores nulos
print("-"*40)
print("Cleaned Dataset:")
print(df_airq.head())
print(df_airq.info())