import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget
import tensorflow as tf
import numpy as np

class IAHandler:
    def __init__(self):
        self.modelo = None

    def cargar_modelo(self):
        self.modelo = tf.keras.models.load_model('mi_modelo.h5') #<<<<<<<<<<<<<< Cambiar el nombre

    def predecir(self, datos_entrada):
        datos = np.array([float(x) for x in datos_entrada.split(',')])
        prediccion = self.modelo.predict(datos.reshape(1, -1))
        return prediccion