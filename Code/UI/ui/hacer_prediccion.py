import sys
import tensorflow as tf
import numpy as np

# Argumentos de la línea de comando (datos de entrada)
datos_entrada = sys.argv[1]

# Convertir los datos de entrada a un formato adecuado
datos = np.array([float(x) for x in datos_entrada.split(',')])

# Cargar el modelo
modelo = tf.keras.models.load_model('mi_modelo.h5') #<<<<<<<<<< Cambiar nombre

# Hacer la predicción
prediccion = modelo.predict(datos.reshape(1, -1))

print(prediccion)
