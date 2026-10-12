# %% [markdown]
# # Árbol de decisión - Empresa de Telecomunicaciones
# 
# - Institución: Instituto Superior de Formación Técnica N° 197
# - Carrera: Técnicatura Superior en Ciencia de Datos e Inteligencia Artificial
# - Cátedra: Procesamiento de Aprendizaje Automático.
# - Docente: Yanina Scudero
# - Alumno: Lucas Nahuel Llanes
# - Año: 2026
# 
# En este trabajo práctico una empresa de telecomunicaciones quiere predecir si un cliente aceptará una oferta de plan de datos móviles. Para ello, se dispone de un conjunto de datos con información de 10 clientes, incluyendo los siguientes atributos:
# 
# - Edad (en años)
# - Nivel de uso mensual de datos (en GB)
# - Tiene línea fija (Sí/No)
# - Aceptó la oferta (Sí/No)
# 
# Partiendo de la consigna anteriormente planteada se procede a analizar los datos suministrados utilizando la biblioteca PANDAS y posteriormente se realizan los cálculos necesarios para obtener la entropía del conjunto original y de los diferentes subconjuntos que se desprenden de los distintos atributos para finalmente seleccionar el mejor de ellos y construir el **árbol de decisión**.
# 
# Este *Notebook* se dividirá en las siguientes secciones:
# 
# - Bibliotecas a utilizar
# - Datos
# - Entropía del conjunto original
# - Atributos a evaluar
#     - Cálculo entropía: edad
#     - Cálculo entropía: tiene línea fija
#     - Cálculo entropía: uso de datos
# - Análisis de ganancia de información
# - Construcción del Árbol de Decisión
# - Conclusiones

# %% [markdown]
# ## Bibliotecas a utilizar

# %%
import pandas as pd
import matplotlib.pyplot as plt
import math

# %% [markdown]
# ## Datos

# %%
# Lista de diccionarios con los datos
datos = [
    {'ID': 1, 'Edad': 24, 'Uso de datos': 2.5, 'Tiene línea fija': 'No', 'Aceptó oferta': 'No'},
    {'ID': 2, 'Edad': 38, 'Uso de datos': 6.0, 'Tiene línea fija': 'Sí', 'Aceptó oferta': 'Sí'},
    {'ID': 3, 'Edad': 29, 'Uso de datos': 3.0, 'Tiene línea fija': 'No', 'Aceptó oferta': 'No'},
    {'ID': 4, 'Edad': 45, 'Uso de datos': 8.0, 'Tiene línea fija': 'Sí', 'Aceptó oferta': 'Sí'},
    {'ID': 5, 'Edad': 52, 'Uso de datos': 7.5, 'Tiene línea fija': 'Sí', 'Aceptó oferta': 'Sí'},
    {'ID': 6, 'Edad': 33, 'Uso de datos': 4.0, 'Tiene línea fija': 'No', 'Aceptó oferta': 'No'},
    {'ID': 7, 'Edad': 41, 'Uso de datos': 5.5, 'Tiene línea fija': 'Sí', 'Aceptó oferta': 'Sí'},
    {'ID': 8, 'Edad': 27, 'Uso de datos': 2.0, 'Tiene línea fija': 'No', 'Aceptó oferta': 'No'},
    {'ID': 9, 'Edad': 36, 'Uso de datos': 6.5, 'Tiene línea fija': 'Sí', 'Aceptó oferta': 'Sí'},
    {'ID': 10, 'Edad': 31, 'Uso de datos': 3.5, 'Tiene línea fija': 'No', 'Aceptó oferta': 'No'}
]

# DataFrame de Pandas
df = pd.DataFrame(datos)

# Gráfico de tabla con Matplotlib
plt.figure(figsize=(10, 4))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

tabla = pd.plotting.table(ax, df, loc='center', cellLoc='center',
                          colWidths=[0.1, 0.1, 0.15, 0.2, 0.2])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.2)

plt.title('Datos de Clientes y Aceptación de Ofertas', pad=20)
plt.show()

# %% [markdown]
# ## Entropía del conjunto original

# %%
# Datos del conjunto
total = 10
aceptaron_oferta = 5
no_aceptaron = 5

# Probabilidades
p_si = aceptaron_oferta / total
p_no = no_aceptaron / total

# Cálculo de entropía
entropy_original = -(p_si * math.log2(p_si) + p_no * math.log2(p_no))
print(f"Entropía del conjunto original: {entropy_original:.3f}")

# %% [markdown]
# ## Función de entropía unificada
# 
# Para evitar errores y repeticiones, se define una única función reutilizable que se usará en todos los atributos.

# %%
def entropy(p1, p2):
    """Calcula la entropía de un subconjunto con dos clases.
    p1 = cantidad de 'Sí', p2 = cantidad de 'No'.
    Devuelve 0 si el subconjunto es puro o vacío."""
    total = p1 + p2
    if total == 0 or p1 == 0 or p2 == 0:
        return 0.0
    prob1 = p1 / total
    prob2 = p2 / total
    return -(prob1 * math.log2(prob1) + prob2 * math.log2(prob2))


def ganancia(entropy_original, subconjuntos):
    """subconjuntos: lista de tuplas (n_si, n_no).
    Calcula la entropía ponderada y devuelve la ganancia de información."""
    total = sum(p1 + p2 for p1, p2 in subconjuntos)
    H_ponderada = sum((p1 + p2) / total * entropy(p1, p2)
                      for p1, p2 in subconjuntos)
    return entropy_original - H_ponderada, H_ponderada

# %% [markdown]
# ## Atributos a evaluar
# - Edad
# - Tiene línea fija
# - Uso de datos

# %% [markdown]
# #### Cálculo entropía: edad
# Se divide el atributo en tres subconjuntos según los siguientes rangos:
# - Joven ≤ 30
# - Adulto 31 - 50
# - Mayor > 50

# %%
# Organización de los datos por rangos de edad
df['Rango Edad'] = pd.cut(df['Edad'],
                          bins=[0, 30, 50, float('inf')],
                          labels=["Joven", "Adulto", "Mayor"],
                          right=False)  # right=False significa ≤ en el límite inferior

print(df[['ID', 'Edad', 'Rango Edad', 'Aceptó oferta']])

print("\nFrecuencias por rango de edad:")
print(df['Rango Edad'].value_counts().sort_index())

print("\nTabla cruzada (Rango Edad vs Aceptó oferta):")
print(pd.crosstab(df['Rango Edad'], df['Aceptó oferta']))

# %%
# Conteos verificados:
# Joven: 0 sí, 3 no
# Adulto: 4 sí, 2 no
# Mayor: 1 sí, 0 no
subconjuntos_edad = [(0, 3), (4, 2), (1, 0)]
ganancia_edad, H_edad = ganancia(entropy_original, subconjuntos_edad)

print(f"Entropía del conjunto original: {entropy_original:.4f}")
print(f"Entropía ponderada (Edad): {H_edad:.4f}")
print(f"Ganancia de información (Edad): {ganancia_edad:.4f}")

# %% [markdown]
# #### Cálculo de entropía: Tiene línea fija

# %%
print("Tabla cruzada (Tiene línea fija vs Aceptó oferta):")
print(pd.crosstab(df['Tiene línea fija'], df['Aceptó oferta']))

# %%
# Conteos verificados:
# Sí: 5 aceptaron, 0 no aceptaron
# No: 0 aceptaron, 5 no aceptaron
subconjuntos_linea = [(5, 0), (0, 5)]
ganancia_linea, H_linea = ganancia(entropy_original, subconjuntos_linea)

print(f"Entropía del conjunto original: {entropy_original:.4f}")
print(f"Entropía ponderada (Tiene línea fija): {H_linea:.4f}")
print(f"Ganancia de información (Tiene línea fija): {ganancia_linea:.4f}")

# %% [markdown]
# #### Cálculo de entropía: Uso de datos
# Se divide el atributo en tres subconjuntos según los siguientes rangos:
# - Bajo ≤ 3
# - Medio 3.1 - 6
# - Alto > 6

# %%
# Organización de los datos por uso de datos
df['Grupos por datos'] = pd.cut(df['Uso de datos'],
                                bins=[0, 3.1, 6.1, float('inf')],
                                labels=["Bajo", "Medio", "Alto"],
                                right=False)  # right=False significa ≤ en el límite inferior

print(df[['ID', 'Uso de datos', 'Grupos por datos', 'Aceptó oferta']])

print("\nFrecuencias por grupo de datos:")
print(df['Grupos por datos'].value_counts().sort_index())

print("\nTabla cruzada (Grupos por datos vs Aceptó oferta):")
print(pd.crosstab(df['Grupos por datos'], df['Aceptó oferta']))

# %%
# Conteos verificados (CORREGIDOS respecto del original):
# Bajo:  0 sí, 4 no
# Medio: 2 sí, 1 no
# Alto:  3 sí, 0 no
subconjuntos_datos = [(0, 4), (2, 1), (3, 0)]
ganancia_datos, H_datos = ganancia(entropy_original, subconjuntos_datos)

print(f"Entropía del conjunto original: {entropy_original:.4f}")
print(f"Entropía ponderada (Uso de datos): {H_datos:.4f}")
print(f"Ganancia de información (Uso de datos): {ganancia_datos:.4f}")

# %% [markdown]
# ## Análisis de Ganancia de Información
# 
# De los cálculos obtenidos:
# 
# | Atributo | Entropía ponderada | Ganancia |
# |----------|--------------------|----------|
# | **Tiene línea fija** | 0.0000 | **1.0000** |
# | **Uso de datos** | 0.2755 | **0.7245** |
# | **Edad** | 0.5510 | **0.4490** |
# 
# El atributo con mayor ganancia es **Tiene línea fija**, con una ganancia perfecta de **1.0**, ya que separa completamente las clases.

# %% [markdown]
# ## Construcción del Árbol de Decisión
# 
# ¿Tiene línea fija?
# 
# ├── Sí → Clase: "Aceptó la oferta"
# 
# └── No → Clase: "No aceptó la oferta"
# 
# Con este árbol simple obtenemos:
# 
# - Precisión: 100% (10/10 casos clasificados correctamente)
# - Complejidad: Mínima (solo 1 atributo necesario)
# - Interpretabilidad: Muy alta
# 
# Este ejemplo representa un caso ideal donde un solo atributo es suficiente para clasificar perfectamente todos los ejemplos.

# %% [markdown]
# ## Conclusiones
# 
# De la información obtenida encontramos que **Tiene Línea Fija** es la mejor división porque su ganancia es perfecta (1.0), lo que separa perfectamente las clases creando dos subconjuntos 100% puros.
# 
# - Subconjunto "Sí": 5 personas de 10
# - Subconjunto "No": 5 personas de 10
# 
# Ambos subconjuntos tienen una entropía de 0, debido a que son puros, lo que hace innecesario dividir aún más porque cada rama clasifica perfectamente, alcanzando la ganancia máxima posible que es 1.0. Esta ganancia perfecta vuelve a **Tiene Línea Fija** como raíz del árbol.