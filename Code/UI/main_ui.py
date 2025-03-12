# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(500, 300)
        MainWindow.setMinimumSize(QSize(500, 300))
        MainWindow.setMaximumSize(QSize(500, 300))
        MainWindow.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.btn_est = QPushButton(self.centralwidget)
        self.btn_est.setObjectName(u"btn_est")
        self.btn_est.setMinimumSize(QSize(0, 130))

        self.gridLayout.addWidget(self.btn_est, 3, 1, 1, 1)

        self.btn_chat = QPushButton(self.centralwidget)
        self.btn_chat.setObjectName(u"btn_chat")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_chat.sizePolicy().hasHeightForWidth())
        self.btn_chat.setSizePolicy(sizePolicy)
        self.btn_chat.setMinimumSize(QSize(50, 130))

        self.gridLayout.addWidget(self.btn_chat, 3, 0, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)
        self.label.setMaximumSize(QSize(16777215, 100))
        font = QFont()
        font.setPointSize(16)
        font.setBold(False)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 2, 0, 1, 2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Athena - Pantalla principal", None))
        self.btn_est.setText(QCoreApplication.translate("MainWindow", u"Ver Estadisticas", None))
        self.btn_chat.setText(QCoreApplication.translate("MainWindow", u"Comenzar Chat", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Athena - Asistente de IA", None))
    # retranslateUi

