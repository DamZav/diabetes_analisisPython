# %% Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display

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

# 3. Comprobamos el cambio y hacemos una copia para trabajar con el DataFrame
df.reset_index(inplace=True)
df.to_csv('db_diabetes.csv', index=False)  # Para que la primera columna se unifique
display(df)

# 4. Cambiamos el index a columna y arreglamos una columna
df = df.rename(columns={'Tipo': 'tipo_diabetes'})
print(df.columns)
display(df)

# 5. Seleccionar solo columnas numéricas
datos_numericos = df.select_dtypes(include=['int64', 'float64'])

# 6. Calcular medidas de localización y variabilidad
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

# 7. Calcular límites para detección de valores atípicos
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

# 8. Identificación de valores atípicos en cada columna
valores_atipicos = {}
for column in datos_numericos.columns:
    valores_atipicos[column] = datos_numericos[(datos_numericos[column] < limite_inferior[column]) |
                                               (datos_numericos[column] > limite_superior[column])][column]
    
# Mostrar valores atípicos por columna
for column, outlier_values in valores_atipicos.items():
    print(f"\nCifras anómalas en {column}:")
    display(outlier_values)

# 9. Calcular y mostrar percentiles específicos
percentiles = [0, 10, 25, 50, 75, 90, 100]
valores_percentiles = datos_numericos.quantile([p / 100 for p in percentiles])
display("Percentiles:")
display(valores_percentiles)

# 10. Gráfico de límites superiores de valores atípicos
limites_atipicos['Límite Superior'].plot(kind='hist', bins=20, title='Límite Superior')
plt.gca().spines[['top', 'right']].set_visible(False)

# 11. Gráfico de comparación de límites
limites_atipicos.plot(kind='scatter', x='Límite Inferior', y='Límite Superior', s=32, alpha=0.8)
plt.gca().spines[['top', 'right']].set_visible(False)

# 12. Gráfico de Salud Pancreática vs Niveles de Enzima Digestiva
plt.figure(figsize=(8, 6))
plt.scatter(df['niveles_de_encimas_digestiva'], df['salud_pancreatica'], alpha=0.7)
plt.xlabel('Niveles de Enzima Digestiva')
plt.ylabel('Salud Pancreática')
plt.title('Salud Pancreática vs Niveles de Enzima Digestiva')

# 13. Interpretación de percentiles
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

# 14. Crear histogramas para cada columna numérica
for column in datos_numericos.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(datos_numericos[column], kde=True, bins=20)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()

# 15. Calcular y mostrar asimetría y curtosis para cada columna numérica
asimetria = datos_numericos.skew()
curtosis = datos_numericos.kurt()

print("Asimetría y Curtosis por columna:")
for column in datos_numericos.columns:
    print(f"\nColumna '{column}':")
    print(f" - Asimetría: {asimetria[column]:.2f}")
    print(f" - Curtosis: {curtosis[column]:.2f}")

# 16. Graficar matriz de correlación
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Seleccionar solo las columnas numéricas
correlation_matrix = numeric_df.corr()

# Graficar el mapa de calor de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# 17. Diagrama de dispersión entre Edad e IMC
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='edad', y='imc', alpha=0.7)  # Ajustar la transparencia con alpha
plt.title("Relación entre Edad e IMC")
plt.xlabel("Edad")
plt.ylabel("IMC")
plt.grid(True)  # Añadir una cuadrícula para mejor visualización
plt.show()

# Entrenar un modelo de Regresión Lineal Simple
# Seleccionar las variables independientes y dependientes
X = df[['edad']]  # Variable independiente (Edad)
y = df['imc']     # Variable dependiente (IMC)

# Dividir los datos en conjuntos de entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo usando los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)  # Error Cuadrático Medio
r2 = r2_score(y_test, y_pred)              # Coeficiente de determinación

# Imprimir las métricas de rendimiento
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R^2): {r2:.2f}")

# 3. Visualizar la regresión lineal
plt.figure(figsize=(10, 6))
# Gráfico de dispersión para los datos reales
sns.scatterplot(x=X_test['edad'], y=y_test, label="Datos reales", alpha=0.7)
# Línea de regresión para las predicciones
sns.lineplot(x=X_test['edad'], y=y_pred, color="red", label="Predicción de regresión lineal")
plt.title("Modelo de Regresión Lineal Simple: Edad vs IMC")
plt.xlabel("Edad")
plt.ylabel("IMC")
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Añadir una cuadrícula
plt.show()
