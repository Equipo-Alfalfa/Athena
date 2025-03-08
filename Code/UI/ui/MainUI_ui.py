# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainUI.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QPushButton,
    QSizePolicy, QWidget)
import mainUI_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(606, 400)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QSize(606, 400))
        icon = QIcon()
        icon.addFile(u":/Icons/arraycolors.ico", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addFile(u"../icons/arraycolors.ico", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        Form.setWindowIcon(icon)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(15)
        font.setWeight(QFont.Light)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setMinimumSize(QSize(150, 0))
        font1 = QFont()
        font1.setPointSize(19)
        self.label.setFont(font1)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.btn_graf = QPushButton(Form)
        self.btn_graf.setObjectName(u"btn_graf")
        sizePolicy2.setHeightForWidth(self.btn_graf.sizePolicy().hasHeightForWidth())
        self.btn_graf.setSizePolicy(sizePolicy2)
        self.btn_graf.setMinimumSize(QSize(0, 130))
        self.btn_graf.setMaximumSize(QSize(600, 200))
        icon1 = QIcon()
        icon1.addFile(u"../../source/icon/sales.ico", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_graf.setIcon(icon1)
        self.btn_graf.setIconSize(QSize(24, 24))

        self.gridLayout_3.addWidget(self.btn_graf, 0, 0, 1, 1)

        self.btn_cln = QPushButton(Form)
        self.btn_cln.setObjectName(u"btn_cln")
        sizePolicy2.setHeightForWidth(self.btn_cln.sizePolicy().hasHeightForWidth())
        self.btn_cln.setSizePolicy(sizePolicy2)
        self.btn_cln.setMinimumSize(QSize(0, 130))
        self.btn_cln.setMaximumSize(QSize(600, 200))
        icon2 = QIcon()
        icon2.addFile(u"../icons/burn.ico", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_cln.setIcon(icon2)
        self.btn_cln.setIconSize(QSize(27, 27))

        self.gridLayout_3.addWidget(self.btn_cln, 1, 1, 1, 1)

        self.btn_busc = QPushButton(Form)
        self.btn_busc.setObjectName(u"btn_busc")
        sizePolicy2.setHeightForWidth(self.btn_busc.sizePolicy().hasHeightForWidth())
        self.btn_busc.setSizePolicy(sizePolicy2)
        self.btn_busc.setMinimumSize(QSize(0, 130))
        self.btn_busc.setMaximumSize(QSize(600, 200))
        icon3 = QIcon()
        icon3.addFile(u"../icons/search2.ico", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_busc.setIcon(icon3)
        self.btn_busc.setIconSize(QSize(24, 24))

        self.gridLayout_3.addWidget(self.btn_busc, 0, 1, 1, 1)

        self.btn_excel = QPushButton(Form)
        self.btn_excel.setObjectName(u"btn_excel")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.btn_excel.sizePolicy().hasHeightForWidth())
        self.btn_excel.setSizePolicy(sizePolicy3)
        self.btn_excel.setMinimumSize(QSize(0, 130))
        self.btn_excel.setMaximumSize(QSize(600, 200))
#if QT_CONFIG(tooltip)
        self.btn_excel.setToolTip(u"<html><head/><body><p>Advertencia:</p><p>Esta operaci\u00f3n podria tardar un tiempo.</p></body></html>")
#endif // QT_CONFIG(tooltip)
        self.btn_excel.setToolTipDuration(-1)
        self.btn_excel.setText(u"Generar Excel con datos de pago")

        self.gridLayout_3.addWidget(self.btn_excel, 1, 0, 1, 1)

        self.btn_raw = QPushButton(Form)
        self.btn_raw.setObjectName(u"btn_raw")

        self.gridLayout_3.addWidget(self.btn_raw, 2, 0, 1, 2)


        self.gridLayout.addLayout(self.gridLayout_3, 3, 0, 1, 1)

#if QT_CONFIG(shortcut)
        self.label.setBuddy(self.label)
#endif // QT_CONFIG(shortcut)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"HR Helper", None))
        self.label_2.setText(QCoreApplication.translate("Form", u" Selecciona una Heramienta a continuaci\u00f3n:", None))
        self.label.setText(QCoreApplication.translate("Form", u"Bienvenido a HR Helper", None))
        self.btn_graf.setText(QCoreApplication.translate("Form", u"Generar Graficos", None))
        self.btn_cln.setText(QCoreApplication.translate("Form", u"Limpiar los datos", None))
        self.btn_busc.setText(QCoreApplication.translate("Form", u"Buscar Trabajadores", None))
        self.btn_raw.setText(QCoreApplication.translate("Form", u"Ver datos sin procesar", None))
    # retranslateUi

