#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__


# ## Generando nuestros datos

# In[18]:


#Parámetros conocidos

val1 = 0.7 #pendiente de la recta
val2 = 0.3 #cruce con x=0

# Creando la linea recta

start = 0
end = 1
step = 0.02 # 50 valores
X = torch.arange(start, end, step).unsqueeze(dim=1) # unsqueeze para insertar una dimension
y = val1 * X + val2 # Los valores de y son las etiquetas (labels)


# In[19]:


X[:10]
#y[:10]


# ## Separando el dataset

# In[20]:


# Separando datos en Train y Test; 80% of data used for training set, 20% for testing
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# In[21]:


print('Forma de datos X_train:',X_train.shape,'\nForma de datos y_train:',y_train.shape,'\nForma de datos X_test:',X_test.shape,'\nForma de datos y_test:',y_test.shape)


# ## Visualizando los datos

# In[22]:


#Funcion para visualizar los datos

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 18)
  plt.ylabel('y', size = 20)
  plt.xlabel('X', size = 20)

  
  plt.scatter(train_data, train_labels, c="b", s=6, label="Training data")
  
 
  plt.scatter(test_data, test_labels, c="g", s=6, label="Testing data")

  if predictions is not None:
    
    plt.scatter(test_data, predictions, c="r", s=6, label="Predictions")


  plt.legend(prop={"size": 18});


# In[23]:


plot_predictions()


# ## Generando un modelo de regresión lineal

# In[24]:


# Class para un modelo de regresion lineal

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # se configuran pesos aleatorios que serán modificados comforme se entrene la red
                                                dtype=torch.float), # float32 determinado
                                   requires_grad=True) 
        
        self.bias = nn.Parameter(torch.randn(1, # se configuran sesgos aleatorios que serán modificados comforme se entrene la red
                                            dtype=torch.float), 
                                requires_grad=True)

    # implementando forward
    def forward(self, x: torch.Tensor) -> torch.Tensor: #x son los datos de entrada
        return self.weights * x + self.bias # formula de regresión lineal


# In[25]:


# parametros de nn.Parameter se inician aleatoriamente; ponemos un seed manual
torch.manual_seed(42)

# Creando una sublcase de nn.Module que contiene los nn.Parameter (s)
model_0 = LinearRegressionModel()

# Listamos los parametros de la subclase creada
list(model_0.parameters())


# In[26]:


# Haciendo predicciones
with torch.inference_mode(): #Evita la modificacion de parametros del modelo
    y_preds = model_0(X_test)


# In[27]:


print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")


# In[28]:


plot_predictions(predictions=y_preds)


# ## Entrenando el modelo

# In[29]:


# Definiendo la función de pérdidas
loss_fn = nn.L1Loss() # L1Loss = mean absolute error (MAE)

# Definiendo el optimizador
optimizer = torch.optim.SGD(params=model_0.parameters(), # parámetros del modelo a optimizar
                            lr=0.01) # tasa de aprendizaje - learning rate


# In[30]:


#Bucles de aprendizaje

torch.manual_seed(42)

epochs = 150 #Definir el número de épocas

# Crear listas de pérdidas vacías para realizar un seguimiento de los valores
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # ENTRENAMIENTO

    
    model_0.train() # modelo en modo entrenamiento

    # 1.Pasar los datos de entrenamiento a través del modelo mediante forward()
    y_pred = model_0(X_train) 

    # 2. Se calculan las pérdidas
    loss = loss_fn(y_pred, y_train)

    # 3. Se establecen en cero los valores del optimizador para comenzar el proceso. Se acumulan en cada época, por lo que hay que establecerlos en 0
    optimizer.zero_grad()

    # 4. Se realiza el backpropagation en la función de pérdidas
    loss.backward() #Calcula el gradiente de la pérdida con respecto a cada parámetro del modelo a actualizar (cada parámetro con requires_grad=True)

    # 5. Se actualizan los valores del modelo. Gradient Descent
    optimizer.step() # Actualiza los parámetros con requires_grad=True con respecto a las pérdidas para mejorarlos.

    ##//////////////
    ### TESTING
    ##//////////////

    # Poner el modelo en modo evaluación
    model_0.eval()

    with torch.inference_mode():  #Modo inferencia para evitar que el modelo "aprenda", no queremos modificar los parametros, más bien usarlos.
      # 1. Pasar los datos de prueba
      test_pred = model_0(X_test)

      # 2. Calcular las péridas en los datos de prueba
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Observamos los datos
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


# In[31]:


#Graficando las pérdidas; convergencia
plt.figure(figsize=(10, 7))
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves", fontsize = 22)
plt.ylabel("Loss", size = 20)
plt.xlabel("Epochs", size = 20)
plt.legend(prop={"size": 18});


# In[32]:


# Viendo los parámetros aprendidos por nuestro modelo

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values are:")
print(f"weight: {val1}, bias: {val2}")


# In[ ]:





# # Haciendo predicciones con el modelo

# In[35]:


model_0.eval()

with torch.inference_mode(): #Modo inferencia para hacer predicciones
    y_preds = model_0(X_test)
y_preds


# In[36]:


plot_predictions(predictions=y_preds)


# ## Guardando y cargando el modelo

# In[38]:


from pathlib import Path

#Creando Directorio
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Directorio del modelo
MODEL_NAME = "modelo_pytorch_regresionlineal.pth" #IMPORTANTE: Los modelos se guardan como .pt o .pth
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Salvando el modelo usando "state dict" 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)


# In[41]:


model_loaded = LinearRegressionModel() #Generamos una nueva instancia del modelo;Los parámetros se establecen aleatoriamente

model_loaded.load_state_dict(torch.load(f=MODEL_SAVE_PATH))# Cargamos el diccionario de parámetros que guardamos previamente en la nueva instancia.


# In[42]:


#Observamos los parámetros guardados, son los mismos que obtuvimos antes de guardarlo
model_loaded.state_dict()


# In[44]:


#Haciendo predicciones

model_loaded.eval() # Ponemos el modelo en modo evaluación

# Modo inferencia para las predicciones
with torch.inference_mode():
    model_loaded_preds = model_loaded(X_test) # perform a forward pass on the test data with the loaded model


# In[45]:


# Comparemos las predicciones del modelo antes de guardarlo y las del modelo cargado. (Deben ser igaules)

y_preds == model_loaded_preds


# In[ ]:




