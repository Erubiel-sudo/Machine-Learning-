#!/usr/bin/env python
# coding: utf-8

# ### ¿Cual es la categoria del nivel de poblacion de una entidad en 2010 observada por primera vez? 
# (Bajo, Medio, Alto)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[69]:


df = pd.read_csv('indicadores.csv', sep=',', encoding='latin-1')
df


# In[3]:


columna_58 = df.iloc[:, 58]
df_y = pd.DataFrame(columna_58, columns=['pobtot_10'])
df_y.index = range(len(df_y))  
display(df_y)


# In[4]:


# spaces in the dataset
datos_nulos = df.isna().sum().sum()
print("Total of null values:", datos_nulos)


# In[5]:


matriz_correlacion = df.cor
columnas_rel = matriz_correlacion['pobtot_10']
correlaciones_ordenadas = columnas_rel.abs().sort_values(ascending=False)
print(correlaciones_ordenadas)


# In[6]:


# Definimos un umbral de 0.7 ya que esto nos indica una correlacion significativa.
umbral_correlacion = 0.7
columnas_interes = correlaciones_ordenadas[correlaciones_ordenadas > umbral_correlacion].index.tolist()

# New dataframe
new_df = df[columnas_interes]
new_df


# In[7]:


null_values = new_df.isnull().sum()
print(null_values)


# In[8]:


#Calculate the mean of the numeric columns
columnas_numericas = new_df.select_dtypes(include=[np.number])
promedios = columnas_numericas.mean()

#Fill the null vallues in the dataset
new_df.loc[:, columnas_numericas.columns] = 
new_df.loc[:, columnas_numericas.columns].fillna(promedios)


# In[9]:


#Comprueba si hay espacios en blanco
null_values = new_df.isnull().sum()
print(null_values)


# In[10]:


new_df


# In[11]:


def categorizar_valor(valor):
    if valor < 10000:
        return "baja"
    elif valor < 100000:
        return "media"
    else:
        return "alta"

new_df['pobtot_10'] = new_df['pobtot_10'].apply(categorizar_valor)
# Definir las variables x e y
y = new_df["pobtot_10"]
x = new_df.drop("pobtot_10", axis=1)
new_df = pd.concat([x, y], axis=1)


# In[12]:


new_df


# In[52]:


y = new_df["pobtot_10"]
x = new_df.drop("pobtot_10", axis=1)


# In[53]:


new_df.columns[0:17]


# In[60]:


X_train = new_df.iloc[:1964, 0:17].values
X_test = new_df.iloc[1964:, 0:17].values

y_train = new_df.iloc[:1964, 17].values
y_test = new_df.iloc[1964:, 17].values


# In[62]:


def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist
 
def predict(x_train, y , x_input, k):
    op_labels = []
     
    for item in x_input: 
         
        #distances storage
        point_dist = []
         
        #data to be processed
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        #preserve the index
        dist = np.argsort(point_dist)[:k] 
         
        labels = y[dist]
         
        #voting
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]
        
        op_labels.append(most_common_label)
 
    return op_labels


# In[57]:


predictions = predict(X_train, y_train, X_test, k=5)
predictions


# In[64]:


accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')


# #### -Implementation of the model with librarie.

# In[72]:


from sklearn.neighbors import KNeighborsClassifier

# Training and test data
X_train = new_df.iloc[:1964, 0:17].values
y_train = new_df.iloc[:1964, 17].values
X_test = new_df.iloc[1964:, 0:17].values
y_test = new_df.iloc[1964:, 17].values

# k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(predictions)
predictions = knn.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')


# #### -The same model with Hamming metric

# In[67]:


X_train = new_df.iloc[:1964, 0:17].values
X_test = new_df.iloc[1964:, 0:17].values

y_train = new_df.iloc[:1964, 17].values
y_test = new_df.iloc[1964:, 17].values

def hamming_distance(p1, p2):
    return np.sum(p1 != p2)

# Función para predecir utilizando KNN con distancia de Hamming
def predict(x_train, y, x_input, k):
    op_labels = []

    for item in x_input:
        point_dist = []
        for j in range(len(x_train)):
            distance = hamming_distance(np.array(x_train[j, :]), item)
            # Calcula la distancia de Hamming
            point_dist.append(distance)
        point_dist = np.array(point_dist)

        dist = np.argsort(point_dist)[:k]
        labels = y[dist]
        # Votación
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]
        op_labels.append(most_common_label)

    return op_labels

# test and trainig data
X_train = X_train.astype(str) 
X_test = X_test.astype(str)
k = 3  
y_pred = predict(X_train, y_train, X_test, k)
#acuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')

