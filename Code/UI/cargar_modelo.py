import tensorflow as tf

def cargar_modelo():
    modelo = tf.keras.models.load_model('mi_modelo.h5') #<<<<<<<<<< Cambiar nombre
    return modelo
