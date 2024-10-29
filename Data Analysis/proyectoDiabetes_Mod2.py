# %%
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt


# 1. Cargar datos diabetes desde un archivo CSV
df = pd.read_csv("./diabetes.csv", encoding='latin-1', index_col=0)
df.head()



# 2. Exploramos BD
print(df.columns.tolist())

def info_general_df(df):
  print('Información general del DataFrame')
  print(f'Cantidad de filas y columnas: {df.shape}')
  print("-"*50)
  print(f'Cantidad de datos nulos por columna: \n{df.isnull().sum()}')
  print("-"*50)
  print(f'Cantidad de datos únicos por columna: \n{df.nunique()}')
  print("-"*50)
  print(f'Tipos de datos por columna: \n{df.dtypes}')

info_general_df(df)



# 3. Hacemos una copia de la base de datos
df_copia = df.copy()


# ======= Limpiar DB =======
# 4. Eliminar columnas que no aportan información necesaria para el enfoque del proyecto
df = df.drop(['Marcadores geneticos',
              'Autoanticuerpos',
              'Factores ambientales',
              'Origen etnico',
              'Niveles de encimas digestiva',
              'Sistomas de inicio temprano'], axis=1) #axis = 1 indica que se eliminan columnas



# 6. Renombro columnas y arreglo una columna
df = df.rename(columns= {
    'Antecedentes familiares': 'antecedentes_familiares', 
    'Niveles de insulina': 'niveles_insulina', 
    'Edad': 'edad', 
    'IMC': 'imc',
    'Actividad fisica ': 'actividad_fisica', 
    'Abitos dieteticos': 'habitos_dieteticos', 
    'Presion sanguinia ': 'presion_sanguinea',
    'Niveles de colesterol': 'niveles_colesterol', 
    'Talla': 'talla',
    'Nivees de glucosa': 'niveles_glucosa',
    'Factores socioecomonicos ': 'factores_socioeconomicos', 
    'Tabaquismo': 'tabaquismo', 
    'Consumo de alcohol': 'consumo_alcohol',
    'Tolerancia a la glucosa': 'tolerancia_glucosa', 
    'Sindrome de ovario poliquístico': 'sindrome_ovario_poliquistico',
    'Diabetes gestional': 'diabetes_gestacional', 
    'Tipo de embarazo': 'tipo_embarazo',
    'Aumento de peso en el embarazo': 'aumento_peso_embarazo', 
    'Salud pancriatica': 'salud_pancreatica',
    'Analisis de orina': 'analisis_orina', 
    'Peso al nacer': 'peso_nacimiento'
})

# Convertir la columna de gramos a kilogramos
df['peso_nacimiento'] = df['peso_nacimiento'] / 1000

# Verifico los primeros valores para asegurar de que la conversión fue correcta
print(df['peso_nacimiento'].head())


#  7. Comprobamos el cambio y hacemos una copia para trabajar con el dataFrame
df.head()
df.reset_index(inplace=True)
df.to_csv('db_diabetes.csv', index=False) #para que la primera columna se unifique
df


# 8. Cambiamos el index a columna y arreglamos una columna
df = df.rename(columns= {'Tipo': 'tipo_diabetes'})
print(df.columns)
display(df)



# ====================== ANÁLISIS ========================

print("-"*50)
print("-"*50)


def conteos(df):
  print('------------------- Estadísticas --------------------')
  print(f'Media de edad: \n{df['edad'].mean()}')
  print("-"*50)
  print(f'El nivel de insulina máxima y mínima son: \n{df['niveles_insulina'].max()}, {df['niveles_insulina'].min()}')
  print("-"*50)
  print(f'El tipo de alcoholismo más común es: \n{df['consumo_alcohol'].mode()[0]}')
  print("-"*50)
  print(f'El nivel de colesterol más alto y más bajo son: \n{df["niveles_colesterol"].max()}, {df["niveles_colesterol"].min()}')

conteos(df)

print("-"*50)
print('¿Hay más fumadores que no fumadores?')

fumadores = df['tabaquismo'].value_counts()['Smoker']
no_fumadores = df['tabaquismo'].value_counts()['Non-Smoker']

if fumadores > no_fumadores:
  print("Hay más fumadores que no fumadores con diabetes.")
else:
  print("No hay más fumadores que noFumadores que padecen de diabetes.")
  print("Puede que la población estudiada de diabetes tenga una mayor prevalencia de no fumadores.")


print("-"*50)
# Calculamos métricas de glucosa

print("Métricas de glucosa:")
avg_glucosa = df['niveles_glucosa'].mean()
glucosa_col = [col for col in df.columns if 'glucosa' in col.lower()]

if glucosa_col:
  avg_glucosa = df[glucosa_col[0]].mean()
else:
  print("Columna 'glucosa' no encontrada en el DataFrame.")

mediana_glucosa = df['niveles_glucosa'].median()
rango_glucosa = df['niveles_glucosa'].max() - df['niveles_glucosa'].min()

print(f"Promedio: {avg_glucosa}, Mediana: {mediana_glucosa}, Rango: {rango_glucosa}")

print("-"*50)

# Calcular correlación entre glucosa y presión arterial

print("Correlación entre glucosa y presión arterial")
correlacion = df['niveles_glucosa'].corr(df['presion_sanguinea'])  
print(correlacion)


# Definir criterios
sobrepeso = df['imc'] > 25
glucosa_alta = df['niveles_glucosa'] > 140

# Filtrar pacientes
pacientes_riesgo = df[sobrepeso & glucosa_alta]

print("-"*50)
# Número de pacientes en riesgo
print(f"Número de pacientes con sobrepeso y glucosa elevada: \n{pacientes_riesgo.shape[0]}")


print("-"*50)

# =================================================================================================


# Contar la frecuencia de cada tipo de diabetes
conteo_tipo_diabetes = df['tipo_diabetes'].value_counts()

# Graficar la distribución de tipo de diabetes
plt.figure(figsize=(12, 6))
conteo_tipo_diabetes.plot(kind='bar', color='purple')
plt.title('Distribución del Tipo de Diabetes')
plt.xlabel('Tipo de Diabetes')
plt.ylabel('Número de Casos')
plt.xticks(rotation=60)
plt.show()


# =================================================================================================


# Calcular percentiles de edades
print('¿Cuál es la distribución de las edades de los pacientes?')
percentiles = df['edad'].describe(percentiles=[0.25, 0.5, 0.75])

# Mostrar percentiles
print(percentiles)
 

# =================================================================================================


# Graficar histograma con 10 bins
plt.figure(figsize=(12, 6))
df['edad'].hist(bins=10, color='orange')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Número de Personas')
plt.show()

# Definir bins y etiquetas para categorizar edades
bins = [20, 30, 40, 50, 60, 70, 80, 90]
labels = ['20-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']

# Crear una nueva columna con los rangos de edad
df['rango_edad'] = pd.cut(df['edad'], bins=bins, labels=labels, right=False)

# Contar cuántas personas hay en cada rango de edad y mostrar el resultado
conteo_rango_edad = df['rango_edad'].value_counts().sort_index()
conteo_rango_edad

# =================================================================================================


# Definir los rangos de edad
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']

# Crear una nueva columna con los rangos de edad
df['rango_edad'] = pd.cut(df['edad'], bins=bins, labels=labels, right=False) # pd.cut() se usa para segmentar y clasificar los datos en bins (intervalos), right=False indica que el intervalo no incluye el límite superior

# Contar la frecuencia de cada tipo de diabetes en cada rango de edad
# grouby() agrupa por rango_edad y tipo_diabetes en función de combinaciones únicas de rangos de edad y tipo de diabetes.
# size() cuenta el número de elementos en cada grupo creado por groupby() = serie con el conteo de casos para cada combinacipon rango_edad y tipo_diabetes
# unstack() trnasforma el índice de la serie en columnas y filas
conteo_tipo_diabetes_por_edad = df.groupby(['rango_edad', 'tipo_diabetes']).size().unstack()


# Graficar la distribución de tipo de diabetes por rango de edad
plt.figure(figsize=(20, 6))
conteo_tipo_diabetes_por_edad.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribución del Tipo de Diabetes por Rango de Edad')
plt.xlabel('Rango de Edad')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Diabetes')
plt.show()

# %%