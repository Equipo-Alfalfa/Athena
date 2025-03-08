import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, QMargins

class ChatBubble(QWidget):
    def __init__(self, text, is_user=True):
        super().__init__()
        self.init_ui(text, is_user)

    def init_ui(self, text, is_user):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Add message text
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        
        # Style the bubble
        if is_user:
            self.setStyleSheet("background-color: #DCF8C6; border-radius: 15px; padding: 10px;")
            layout.setContentsMargins(QMargins(50, 10, 10, 10))
        else:
            self.setStyleSheet("background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 15px; padding: 10px;")
            layout.setContentsMargins(QMargins(10, 10, 50, 10))
        
        layout.addWidget(message_label)

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Ventana de Chat Estilizada")
        self.setGeometry(300, 300, 400, 500)
        
        # Scroll area for chat
        scroll_area = QScrollArea()
        self.setCentralWidget(scroll_area)

        # Container widget
        container = QWidget()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)

        # Layout for the container
        self.chat_layout = QVBoxLayout(container)
        self.chat_layout.addStretch(1)

        # Add some example messages
        self.add_message("¡Hola! ¿Cómo estás?", False)
        self.add_message("¡Hola! Estoy bien, gracias. ¿Y tú?", True)
        self.add_message("¡Muy bien también! ¿Qué has estado haciendo?", False)

    def add_message(self, text, is_user):
        chat_bubble = ChatBubble(text, is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, chat_bubble)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ChatWindow()
    main_window.show()
    sys.exit(app.exec())