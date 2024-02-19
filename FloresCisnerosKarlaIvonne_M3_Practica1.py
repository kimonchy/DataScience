#!/usr/bin/env python
# coding: utf-8

# # Práctica 1
# 
# ## Objetivo:
# 
# El participante reunirá todas las nociones estadísticas adquiridas en el Módulo, para responder 
# algunas preguntas formuladas sobre un conjunto de datos a analizar.
# 
# ### Elaborado por Karla Ivonne Flores Cisneros el 18 de febrero de 2024
# 
# ### Módulo 3 Diplomado de Ciencia de Datos: Estadística y probabilidad para ciencia de datos
# 
# ###### Temas selectos de estadística: Aplicación en un modelo de regresión lineal
# 
# 
# La estructura del modelo de regresión lineal simple es $y=\beta_{0}+\beta_{1}x_1 +\epsilon$, donde los coeficientes $\beta_{0}$ y $\beta_{1}$, son parámetros del modelo denominados coeficientes de regresión.  Podemos usar la información proporcionada por una muestra para hallar estimadores (${\hat{\beta}_j}$) de estos.
# 
# Esto significa que, el modelo generado, es una estimación de la relación poblacional a partir de la relación que se observa en la muestra y, por lo tanto, está sujeta a variaciones. Para cada uno de los coeficientes de la ecuación de regresión lineal ($\beta_j$) se puede realizar una prueba de hipótesis para evaluar su significancia. La primera estadística  empleada para validar su significancia proviene de la prueba $t$.
# 
# **Hipótesis**
# 
# $H_0: \beta_j = 0$ vs $H_{1}: \beta_j \neq 0$
# 
# 
# Bajo $H_0$, el predictor  $x_j$  no contribuye al modelo, en presencia del resto de predictores. En el caso de regresión lineal simple, se puede interpretar también como que no existe relación lineal entre ambas variables por lo que (la ordenada) y la  pendiente del modelo son cero.
# 
# **Estadística de prueba**
# 
# $$t = \frac{\hat{\beta}_j}{se(\hat{\beta}_j)}$$
# 
# 
# donde
# 
# 
# $$ SE(\hat{\beta}_j)^2 = {\sqrt{\frac{s^2}{S_{xx}}}} \quad \text{con} \quad s^2=\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{n-2} \quad \text{y} \quad S_{xx}=\sum_{i=1}^{n}\big(x_{i}-\overline{x}\big)^2;$$ 
# 
# 
# 
# 1. Región crítica:
# 
# $|t| \geq t_{1-\alpha/2,n-2}$, con $t_{1-\alpha/2,n-2}$ el cuantil (superior) $\alpha/2$ de una distribución $t$ de Student con $n-2$ grados de libertad. Notar que $\alpha$ es el nivel de significancia deseado para la prueba, interpretado como la probabilidad de rechazar $H_{0}$ cuando es verdadera (error Tipo I). 
# 
# 2. $p$-valor:
# 
# Es el nivel mínimo de significancia tal que se rechaza la hipótesis nula. En este caso,
# 
#  $$ p-\text{valor}=P(|t|>t_{1-\alpha/2,n-2})$$
# 	
#     
#    **Regla de decisión**
#     
#    1. Rechazar $H_{0}$ si $|t| \geq t_{1-\alpha/2,n-2}$, o bien
#    
#    2. Rechazar $H_{0}$, si el $p$-valor es menor o igual que el nivel de significancia $\alpha$.
# 
# 
# ### Significancia del modelo: Prueba F
# 
# 
# El análisis de varianza es una herramienta que sirve para probar la significancia del modelo de regresión. Este contraste responde a la pregunta de si el modelo en su conjunto es capaz de predecir la variable respuesta mejor de lo esperado por azar, o lo que es equivalente, si al menos uno de los predictores que forman el modelo contribuye de forma significativa.
# 
# 
# $$\sum_{i=1}^{n}(y_{i}-\hat{y}) = \sum_{i=1}^{n}(\hat{y}_{i}-\hat{y})^2+\sum_{i=1}^{n}(y_{i}-\hat{y})^2$$
# 
# $$ SCT= SCR + SCE$$
# 
# 
# donde $SCT$ denota la suma de cuadrados (total) respecto a la media, SCR es la suma de cuadrados de la regresión y SCE la suma de cuadrados del error (o residuos). La primera se interpreta como la variabilidad total, mientras que la segunda es la variabilidad explicada por la regresión y el último término, se conoce como variabilidad no explicada.   
# 	
# 
# 
# **Estadística de prueba**
# 
# Una observación importante es, si la variabilidad explicada es pequeña, entonces la recta de regresión no condensa bien la variabilidad de los datos. Y aunque, en general, no podemos comparar la variabilidad explicada sobre la no explicada, puesto que no conocemos su distribución; es posible mostrar que si $\beta_{1}=0$, entonces la estadística de prueba es
# 
# 
# $$\frac{SCR}{SCE/(n-2)}=	\frac{\sum_{i=1}^{n}(\hat{y}_{i}-\hat{y})^2}{\sum_{i=1}^{n}(y_{i}-\hat{y})^2/(n-2)} \sim F_{1,n-2}$$
# 
# 
# donde $F_{1,n-2}$ denota a la distribución $F$ con 1 y $n-2$ grados de libertad. Esto da una guía para constrastar la hipótesis nula $H_{0}:\beta_{1}$, pero ahora desde un enfoque de análisis de varianza, comparando a través de un cociente la variabilidad explicada por la regresión y la variabilidad explicada por el error.
# 	
# Así, podemos considerar la prueba $F$, basada pecisamente en la distribución de probabilidad $F$, para contrastarla significancia en la regresión. Cuando sólo se tiene una variable independiente, como es el caso de regresión lineal simple, la prueba $F$ lleva a la misma conclusión que la prueba $t$ sobre el coeficiente $\beta_{1}$; es decir, si la prueba $t$ indica que $\beta_{1} \neq 0$, puede existir una relación lineal significante. Asimismo, constrastando la misma hipótesis, la prueba $F$ también indicará que hay la posibilidad de exista una relación lineal significante. Pero cuando hay más de una variable independiente, sólo la prueba $F$ puede usarse para probar que existe una relación significativa en general.	
# 	
# **Regla de decisión**
# 	
# Entonces, bajo el supuesto de la hipótesis nula $H_{0}:\beta_{1}=0$ cierta, esta técnica proporciona una ruta para calcular el estadístico de prueba, denotado como $F$, cuya regla de decisión nos indica compararlo con el cuantil $1-\alpha$ de una distribución $F$ con $(1,n-2)$ grados de libertad. Equivalente a contrastar respecto al valor $F^{\alpha}_{1,n-2}$ que delimita el área $(1-\alpha)\cdot 100 \%$ en la cola, donde la regla es rechazar $H_{0}$ si $F> F^{\alpha}_{1,n-2}$.  En cambio,  aplicando el $p$-valor, se rechaza la hipótesis nula si el $p$-valor es menor o igual al valor de $\alpha$. 
# 
# Veamos un ejemplo práctico.
# 
# 

# In[17]:


#!pip install pandas


# In[18]:


#!pip install seaborn


# In[19]:


#!pip install scikit-learn


# In[20]:


#!pip install statsmodels


# In[21]:


# Creamos el ambiente para el análisis de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# ### Generación de los datos
# 
# El conjunto de datos Birthweight contiene la información de 42 bebés al nacer. La pregunta de investigación es saber si existe
# una relación entre al peso al nacer y el tiempo de gestación. La variable dependiente es Peso al nacer, dada en libras, y la 
# variable independiente para esta actividad es la edad gestacional de cada bebé al nacer, en semanas.

# In[23]:


# Definimos los datos
#==============================================================================
datosNP = pd.read_csv('Birthweight.csv')

datosNP


# ## El modelo de regresión desea explicar las carreras si existe una relación entre al peso al nacer y el tiempo de gestación.
# 
# a ) Empezamos con la observación de la gráfica de los datos en forma de dispersión de puntos, donde de forma general podemos ver una aparente relación donde a mayor número de semanas en desarrollo gestacional, el peso en libras del recién nacido parece ser mayor conforme avanza en el eje de tiempo de gestación. Si marcaramos una línea que trazara la unión de dichos puntos, pudieramos apreciar una relación lineal positiva solo a primera vista. Donde la variable _dependiente_ peso del bebé, es mínima en 1.92 del punto en semanas de gestación 33 semanas y 4.57 en 41 semanas.

# In[48]:


# Gráfica semanas gestacionales vs. peso en libras
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

datosNP.plot(
    x    = 'Gestation',
    y    = 'Birthweight',
    c    = 'darkmagenta',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Gráfica de semanas gestacionales vs. peso en libras ');

datosNP.describe()


# In[40]:


gestation = datosNP["Gestation"]
birthweight = datosNP["Birthweight"]

pearsons_coefficient = np.corrcoef(gestation, birthweight)

print("El coeficiente de correlación de Pearson es: \n" ,pearsons_coefficient)


# In[43]:


# División de los datos en train y test
# ==============================================================================
X = datosNP['Gestation']
y = datosNP['Birthweight']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


# In[44]:


# Creación del modelo utilizando el modo fórmula (parecido a la forma de hacerlo en R)
# ==============================================================================
datos_train = pd.DataFrame(np.hstack((X_train, y_train)), columns=['gestation', 'birthweight'])  
modelo = smf.ols(formula = 'birthweight ~gestation', data = datos_train)
modelo = modelo.fit()
print(modelo.summary())


# ## Estimadores: Coeficiente de Correlación, R^2, P-Value, ANOVA
# 
# b) Basándome en el análisis de regresión presentado:
# 
# I) Un coeficiente de _correlación de Pearson_ de $0.70830289$ indica una correlación bastante fuerte entre las variables que están siendo correlacionadas. Según los criterios comunes de interpretación:
# 
# ∣r∣<0.3: Correlación débil
# 0.3≤∣r∣<0.7: Correlación moderada
# ##### ∣r∣≥0.7: Correlación fuerte
# 
# II) El valor de _R-cuadrado (R-squared)_ es $0.567$, lo que indica que aproximadamente el $56.7%$ de la variabilidad en la variable de respuesta _birthweight_ puede ser explicada por la variable independiente _gestation_. Esto sugiere que el modelo de regresión explica una cantidad considerable de la variabilidad en la variable de respuesta.
# 
# III) El coeficiente para la variable independiente gestation es $0.1723$, lo que sugiere que, en promedio, por cada unidad de aumento en la gestación (en semanas), el peso al nacer _birthweight_ aumenta en 0.1723 unidades.
# 
# IV) ANOVA, p-valor e Intercepto: 
# 
# Se obtuvo una _estadística ANOVA_ $F=40.63$ con un p-valor  $4.25e-07$ muy cercano a cero y que es menor que 5% o 0.05. Del análisis de varianza se tiene evidencia para rechazar la hipótesis nula de que todos los coeficientes sean cero. El _intercepto_ es $-3.4746$ (a 3 grados de libertad). Esto puede interpretarse como el valor esperado del peso al nacer cuando la gestación es cero semanas. 
# 
# ## Intervalos de Confianza
# 
# c) El modelo ajustado es $y=\beta_0 + \beta_1 x = -3.4746 + 0.1723x$ y los intervalos del 95% de confianza son:
# 
# [-5.636 , -1.313] para $\beta_0$ y [0.117 , 0.227] para $\beta_1$
# 
# ## Pruebas de hipótesis
# 
# d) En el análisis de regresión lineal:
# 
# I) Prueba de hipótesis para el intercepto:
# **Hipótesis nula (H0):** El intercepto (β0) es igual a cero.
# **Hipótesis alternativa (H1):** El intercepto (β0) no es igual a cero.
# **Resultado:** El valor p asociado con el intercepto es 0.003, lo que indica que el intercepto es estadísticamente significativo a un nivel de significancia del 0.05. Por lo tanto, rechazamos la hipótesis nula y concluimos que el intercepto es significativamente diferente de cero.
# 
# II) Prueba de hipótesis para la variable _gestation_ semanas de gestación:
# **Hipótesis nula (H0):** El coeficiente de gestation (β1) es igual a cero.
# **Hipótesis alternativa (H1):** El coeficiente de gestation (β1) no es igual a cero.
# **Resultado:** El valor p asociado con la variable gestation es 4.25e-07, lo que indica que la gestation es estadísticamente significativa a un nivel de significancia del 0.05. Por lo tanto, rechazamos la hipótesis nula y concluimos que la gestation tiene un efecto significativo en la variable de respuesta birthweight.
# 
# III) Prueba de hipótesis para la significancia global del modelo:
# **Hipótesis nula (H0):** Todos los coeficientes del modelo son iguales a cero, es decir, el modelo no tiene poder predictivo.
# **Hipótesis alternativa (H1):** Al menos uno de los coeficientes del modelo no es igual a cero, es decir, el modelo tiene poder predictivo.
# **Resultado:** La estadística F es 40.63 con un valor p asociado de 4.25e-07, lo que indica que el modelo de regresión en su conjunto es estadísticamente significativo a un nivel de significancia del 0.05. Por lo tanto, rechazamos la hipótesis nula y concluimos que el modelo tiene poder predictivo.

# ## Conclusiones
# 
# e) A partir del análisis de regresión, se sugiere que la duración de la gestación (variable independiente) tiene un efecto significativo en el peso al nacer (como dependiente), y el modelo de regresión parece ser útil para predecir el peso al nacer basado en la duración de la gestación, entonces el modelo de regresión lineal es factible

# Referencia: Amat R. (2021). Estadistica-machine-learning-python. https://github.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python#estadistica-machine-learning-python. 
