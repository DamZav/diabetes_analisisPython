# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Cargar datos diabetes desde un archivo CSV
df = pd.read_csv("./Proyecto/diabetes.csv", encoding='latin-1', index_col=0)
df.head()


# 2. Exploramos BD para limpiar
print(df.columns)


# 3. Hacemos una copia de la base de datos
df_copia = df.copy()


# ======= Limpiar DB =======
# 4. Eliminar columnas que no aportan información
df = df.drop(['Marcadores geneticos',
              'Autoanticuerpos',
              'Factores ambientales',
              'Origen etnico',
              'Niveles de encimas digestiva',
              'Sistomas de inicio temprano'], axis=1) #axis = 1 indica que se eliminan columnas


# 5. Exploramos columnas que pueden tener NaN
nully = df.isna().sum()
# df[df.isna().any(axis=1)]
# print(nully) # Ninguna celda tiene datos nulos


# 6. Renombro columnas
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

df.head()



df.reset_index(inplace=True)
df.to_csv('db_diabetes.csv', index=False) #para que la primera columna se unifique
df.to_csv('db_diabetes2222.csv', index=False)
df
# print(df.columns)


# # Contar la frecuencia de antecedentes familiares
# conteo_antecedentes = df['antecedentes_familiares'].value_counts()

# # Graficar
# conteo_antecedentes.plot(kind='bar', color='skyblue')
# plt.title('Distribución de Antecedentes Familiares')
# plt.xlabel('Antecedentes Familiares')
# plt.ylabel('Número de Pacientes')
# plt.xticks(rotation=0)
# plt.show()

# conteo_antecedentes = df['antecedentes_familiares'].value_counts()
# print(conteo_antecedentes)

# df.head()






# ===============================================
# Visualize the distribution of target variable (types of diabetes)
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='Target')
# plt.title('Distribution of Diabetes Types')
# plt.xlabel('Diabetes Type')
# plt.ylabel('Count')
# plt.show()



# %%