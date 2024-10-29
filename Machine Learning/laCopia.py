# %% Importar librerías necesarias
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para gráficos
sns.set_theme(style="whitegrid")

# 1. Cargar datos desde un archivo CSV
ruta_dataset = "./diabetes.csv"
df = pd.read_csv(ruta_dataset, encoding='latin-1', index_col=0)
display(df.head())  # Mostrar las primeras filas del DataFrame

# 2. Análisis exploratorio de datos
# Verificar valores faltantes
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Eliminar filas y columnas con valores nulos y duplicados
df = df.dropna()  # Elimina filas con valores nulos
df = df.dropna(axis=1)  # Elimina columnas con valores nulos
df = df.drop_duplicates()  # Elimina filas duplicadas
print("-"*50)
display(df.columns)


# Renombro columnas y arreglo una columna
df = df.rename(columns= {
    'Marcadores geneticos': 'marcadores_geneticos',
    'Autoanticuerpos': 'autoanticuerpos',
    'Antecedentes familiares': 'antecedentes_familiares', 
    'Factores ambientales': 'factores_ambientales',
    'Niveles de insulina': 'niveles_insulina', 
    'Edad': 'edad', 
    'IMC': 'imc',
    'Actividad fisica ': 'actividad_fisica', 
    'Abitos dieteticos': 'habitos_dieteticos', 
    'Presion sanguinia ': 'presion_sanguinea',
    'Niveles de colesterol': 'niveles_colesterol', 
    'Origen etnico': 'origen_etnico',
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
    'Niveles de encimas digestiva': 'niveles_de_encimas_digestiva',
    'Analisis de orina': 'analisis_orina', 
    'Peso al nacer': 'peso_nacimiento',
    'Sistomas de inicio temprano': 'sintomas_de_inicio_temprano'
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


# Seleccionar solo columnas numéricas
datos_numericos = df.select_dtypes(include=['int64', 'float64'])

# 3. Calcular medidas de localización y variabilidad
valores_media = datos_numericos.mean()
valores_mediana = datos_numericos.median()
desviacion_estandar = datos_numericos.std()
iqr = datos_numericos.quantile(0.75) - datos_numericos.quantile(0.25)

# Crear un DataFrame con estadísticas descriptivas
estadisticas_descriptivas = pd.DataFrame({
    'Media': valores_media,
    'Mediana': valores_mediana,
    'Desviación Estándar': desviacion_estandar,
    'IQR': iqr
})
display("Estadísticas Descriptivas:")
display(estadisticas_descriptivas)

# 4. Calcular límites para detección de valores atípicos
Q1 = datos_numericos.quantile(0.25)
Q3 = datos_numericos.quantile(0.75)
limite_inferior = Q1 - 1.5 * iqr
limite_superior = Q3 + 1.5 * iqr

# Crear un DataFrame con los límites de valores atípicos
limites_atipicos = pd.DataFrame({
    'Límite Inferior': limite_inferior,
    'Límite Superior': limite_superior
})
display("Límites atípicos:")
display(limites_atipicos)

# 5. Gráfico de límites superiores de valores atípicos
limites_atipicos['Límite Superior'].plot(kind='hist', bins=20, title='Límite Superior')
plt.gca().spines[['top', 'right']].set_visible(False)

# 6. Gráfico de comparación de límites
limites_atipicos.plot(kind='scatter', x='Límite Inferior', y='Límite Superior', s=32, alpha=0.8)
plt.gca().spines[['top', 'right']].set_visible(False)

# 7. Identificación de valores atípicos en cada columna
valores_atipicos = {}
for column in datos_numericos.columns:
    valores_atipicos[column] = datos_numericos[(datos_numericos[column] < limite_inferior[column]) |
                                               (datos_numericos[column] > limite_superior[column])][column]
    
# Mostrar valores atípicos por columna
for column, outlier_values in valores_atipicos.items():
    print(f"\nCifras anómalas en {column}:")
    display(outlier_values)

# 8. Calcular y mostrar percentiles específicos
percentiles = [0, 10, 25, 50, 75, 90, 100]
valores_percentiles = datos_numericos.quantile([p / 100 for p in percentiles])
display("Percentiles:")
display(valores_percentiles)

# 9. Gráfico de Salud Pancreática vs Niveles de Enzima Digestiva
plt.figure(figsize=(8, 6))
plt.scatter(df['niveles_de_encimas_digestiva'], df['salud_pancreatica'], alpha=0.7)
plt.xlabel('Niveles de Enzima Digestiva')
plt.ylabel('Salud Pancreática')
plt.title('Salud Pancreática vs Niveles de Enzima Digestiva')

# 10. Interpretación de percentiles
print("Interpretación de los percentiles:")
for column in datos_numericos.columns:
    print(f"\nPara la columna '{column}':")
    for p in percentiles:
        valor = valores_percentiles[column][p / 100]
        if p == 50:
            print(f" - Percentil {p}: la mediana, con valor {valor}. La mitad de los datos está por debajo de este valor.")
        elif p < 50:
            print(f" - Percentil {p}: {valor}. Aproximadamente el {p}% de los datos están por debajo de este valor.")
        else:
            print(f" - Percentil {p}: {valor}. Aproximadamente el {100 - p}% de los datos están por encima de este valor.")

# 11. Crear histogramas para cada columna numérica
for column in datos_numericos.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(datos_numericos[column], kde=True, bins=20)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()

# 12. Calcular y mostrar asimetría y curtosis para cada columna numérica
asimetria = datos_numericos.skew()
curtosis = datos_numericos.kurt()

print("Asimetría y Curtosis por columna:")
for column in datos_numericos.columns:
    print(f"\nColumna '{column}':")
    print(f" - Asimetría: {asimetria[column]:.2f}")
    print(f" - Curtosis: {curtosis[column]:.2f}")



fig = px.histogram(df, x="niveles_colesterol", nbins=100)
fig.show()

sns.histplot(data=df, x='niveles_glucosa', bins=50)

# Explorar la frecuencia de variables categóricas usando value_counts y moda
categorical_vars = ['marcadores_geneticos', 'autoanticuerpos',
                    'antecedentes_familiares', 'factores_ambientales',
                    'actividad_fisica', 'habitos_dieteticos']

# Calcular y mostrar las frecuencias
frequency_counts = {var: df[var].value_counts() for var in categorical_vars}
for var, counts in frequency_counts.items():
    print(f"Frecuencia de {var}:\n{counts}\n")

# Calcular la moda de cada variable categórica
mode_values = {var: df[var].mode()[0] for var in categorical_vars}
print("Moda de cada variable categórica:")
print(mode_values)

# Generar tablas de contingencia
contingency_table = pd.crosstab(df['tipo_diabetes'], df['autoanticuerpos'])
print("\nTabla de contingencia entre 'Tipo' y 'Autoanticuerpos':")
print(contingency_table)

# Seleccionar solo las columnas numéricas
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calcular la matriz de correlación
correlation_matrix = numeric_df.corr()

# Graficar el mapa de calor de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor de la Matriz de Correlación")
plt.show()


# diagrama de dispersión entre Edad e IMC
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='edad', y='imc')
plt.title("Relación entre Edad e IMC")
plt.xlabel("Edad")
plt.ylabel("IMC")
plt.show()


#entrenar un modelo de Regresión Lineal Simple con las variables Edad (como variable independiente) e IMC (como variable dependiente)
# Seleccionar las variables
X = df[['edad']]  # Variable independiente (Edad)
y = df['imc']     # Variable dependiente (IMC)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R^2): {r2:.2f}")

# Visualizar la regresión lineal
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['edad'], y=y_test, label="Datos reales")
sns.lineplot(x=X_test['edad'], y=y_pred, color="red", label="Predicción de regresión lineal")
plt.title("Modelo de Regresión Lineal Simple: Edad vs IMC")
plt.xlabel("Edad")
plt.ylabel("IMC")
plt.legend()
plt.show()