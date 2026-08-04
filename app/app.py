# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
# Created by: PyQt5 UI code generator 5.9.2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap

from main import *

import sys
import os
import re
from os.path import splitext, exists, join

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 600)
        
        self.img_path = None
        # Use absolute path to ensure it works regardless of working directory
        self.models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyTorch_YOLOv3", "checkpoints"))
        print("Looking for models in:", self.models_path)

        self.models = []
        if not os.path.exists(self.models_path):
            print("Folder not found:", self.models_path)
        else:
            for model_file in os.listdir(self.models_path):
                name, ext = os.path.splitext(model_file)
                if ext == ".pth" or ext == ".pt":
                    self.models.append(name)
            print("Found models:", self.models)

        self.models = sorted(self.models)
        self.current_model = None
        self.Dice_list = None 
        self.bone_num = None

        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 440, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.GetImage)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 440, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.StartSegmentation)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 140, 161, 271))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 140, 161, 271))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 50, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(250, 50, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(410, 140, 161, 271))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(600, 140, 161, 271))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(450, 50, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(620, 50, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

    def StartSegmentation(self):

        if not self.img_path:
            print("please choose image first")
            return

        delete_valid_data(r"./valid_data")
        delete_valid_data(r"./coordinate")

        if self.current_model:
            detect_one(self.img_path, self.current_model)
        else:
            detect_one(self.img_path)

        self.Dice_list, self.bone_num = Segmentation_one(self.img_path[-4:] + ".png")
        self.FindImage()

        for i, dice in enumerate(self.Dice_list):
            line_edit = getattr(self, f"lineEdit_{i+1}", None)
            if line_edit:
                line_edit.setText(str(dice if dice is not None else 0))

        label_path = f"./source/label/{self.img_path[-4:]}.png"
        num = self.connected_component_label(label_path)
        total_dice = sum(d for d in self.Dice_list if d is not None)
        avg = round(total_dice / self.bone_num, 2)

        self.lineEdit_average.setText(str(avg))
        self.lineEdit_original.setText(str(num))
        self.lineEdit_num.setText(str(self.bone_num))

    def GetImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"./source/image", "Image files (*.png *.jpg *.jpeg *.JPG *.PNG *.JPEG)")
        self.label.setPixmap(QPixmap(file_path))
        self.label.setScaledContents(True)
        self.img_path = splitext(file_path)[0].replace("/home/p76094266/source/image/", "")
        self.clear()
        
    def clear(self):
        for i in range(1, 21):
            line_edit = getattr(self, f"lineEdit_{i}", None)
            if line_edit:
                line_edit.setText("0")
        self.lineEdit_average.setText("0")
        self.lineEdit_num.setText("0")
        self.lineEdit_original.setText("0")

    # (Other methods remain unchanged)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
