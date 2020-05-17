# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:42:34 2020

@author: Bassmala
"""
import sys
from PyQt5 import QtCore, QtWidgets , QtGui
import math
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
from CV404Frequency import Filter
from CV404Filters import filters
from CV404Histograms import Histogram
from CV404SIFT import SIVT
from Harris import Harris
from CV404TemplateMatching import TM
from CVHough import Hough
from CV404ActiveContour import ActiveContour
from gui import Ui_MainWindow

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.comboBox.addItem("Average")
        self.ui.comboBox.addItem("Median")
        self.ui.comboBox.addItem("Gaussian")
        self.ui.comboBox.addItem("Edge detection")
        self.layout()
    def layout(self): 
       
        self.ui.pushButton_histograms_load_2.clicked.connect(lambda: Filter.load_first_image(self))
        self.ui.pushButton_histograms_load_3.clicked.connect(lambda:Filter.load_second_image(self))
        self.ui.pushButton_histograms_load_4.clicked.connect(lambda:Filter.make_hybrid(self))
        self.ui.pushButton_filters_load.clicked.connect(lambda:filters.button_clicked(self))
        self.ui.comboBox_2.currentIndexChanged.connect(lambda:filters.noise_combobox_changed(self))
        self.ui.comboBox.currentIndexChanged.connect(lambda:filters.filter_combobox_changed(self))
        
        self.ui.comboBox_3.currentIndexChanged.connect(lambda:filters.edge_combobox_changed(self))
        self.ui.pushButton_histograms_load.clicked.connect(lambda:Histogram.load_image(self))
        self.ui.equalization.clicked.connect(lambda:Histogram.input_histogram(self))
        self.ui.coloredhistogram.clicked.connect(lambda:Histogram.colored_histogram_button_pressed(self))
        
        self.ui.Hough_load_4.clicked.connect(lambda: Harris.get_corners(self))
        self.ui.Hough_load_3.clicked.connect(lambda: Harris.load_image(self))

        self.ui.Hough_load.clicked.connect(lambda: Hough.load_image(self)) 
        self.ui.Hough_load_2.clicked.connect(lambda: Hough.apply_button_clicked(self,self.event)) 
        
        self.ui.Hough_load_5.clicked.connect(lambda: ActiveContour.button_clicked(self)) 
        self.ui.Input_img_4.mousePressEvent= self.getPos
        self.ui.apply.clicked.connect(lambda: ActiveContour.apply_clicked(self))
        self.ui.clear.clicked.connect(lambda: ActiveContour.clear_clicked(self))
        
        self.ui.Load_imageA_2.clicked.connect(lambda: SIVT.load_first_image(self))
        self.ui.Load_imageB_2.clicked.connect(lambda: SIVT.load_second_image(self))
        self.ui.pushButton_match_4.clicked.connect(lambda: SIVT.match_btn_pressed(self))
        
        
        self.ui.Load_imageA.clicked.connect(lambda: TM.load_first_image(self))
        self.ui.Load_imageB.clicked.connect(lambda: TM.load_second_image(self))
        self.ui.pushButton_match.clicked.connect(lambda: TM.matching_pressed(self))
        
        
        
        
        
    def getPos(self,event) :
           
        self.x=math.floor((event.pos().x()*self.pixmap.width())/self.ui.Input_img_4.frameGeometry().width())
        print (self.x)
        self.y=math.floor((event.pos().y()*self.pixmap.height())/self.ui.Input_img_4.frameGeometry().height())
        print(self.y)    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()        
        