import sys

from PySide6.QtWidgets import (QApplication, QMainWindow,
QPushButton)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Cargar el archivo .ui
        loader = QUiLoader()
        file = QFile("MainUi.ui")
        file.open(QFile.ReadOnly)
        
        # Cargar la interfaz en la ventana principal
        self.ui = loader.load(file, self)
        file.close()


        # Establecer la interfaz cargada como la ventana central
        self.setCentralWidget(self.ui)

        # Conectar los botones a sus funciones
        self.setup_connections()

    def setup_connections(self):
        pass
    def btn_chat_clicked(self):
        pass
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
