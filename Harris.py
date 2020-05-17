# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:36:39 2020

@author: Bassmala
"""

from PyQt5 import QtCore, QtWidgets , QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
from PyQt5.QtCore import QSize,pyqtSlot,QTimer,QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import qimage2ndarray
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage 
import math
import sys
import os
import cv2
from scipy import signal as sig
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage as ndi
from PIL import Image
from gui import Ui_MainWindow

class Harris:
    
    def load_image(self):  
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            self.colored_img = io.imread(self.path)
            self.gray_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.image1 = QPixmap(fileName)
            Harris.Display(self.gray_img,self.ui.Input_img_3)
            
          
            
            self.ui.label_30.setText(os.path.basename(fileName))
            self.ui.label_31.setText(str(self.image1.height()) +"X"+str(self.image1.width()))    
        
        
          
    def Display( img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image) 
        
        
        
        
        
        
        
    def gradient_x(imggray):
        ##Sobel operator kernels.
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        return sig.convolve2d(imggray, kernel_x, mode='same')
    def gradient_y(imggray):
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return sig.convolve2d(imggray, kernel_y, mode='same')
    
    
    
    
    def get_harris_response(imggray,k):
        I_x = Harris.gradient_x(imggray)
        I_y = Harris.gradient_y(imggray)
        Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
        Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
        Iyy = ndi.gaussian_filter(I_y**2, sigma=1)
        detA = Ixx * Iyy - Ixy ** 2
        # trace
        traceA = Ixx + Iyy
        
        harris_response = detA - k * traceA ** 2
        return harris_response 
    
    
    def thresholding(threshold,harris_res,img_copy_for_corners):        
    
        for rowindex, point in enumerate(harris_res):
            for colindex, r in enumerate(point):
                if np.abs(r) > threshold * np.max(r):
                    # this is a corner
                    img_copy_for_corners[rowindex, colindex] = [255,0,0]
    
        return img_copy_for_corners
        
    
    def get_corners(self):
        k=0.05
        threshold=2
        threshold= int(self.ui.textEdit_4.toPlainText())
        k = float(self.ui.textEdit_5.toPlainText())
        harris_response=Harris.get_harris_response(self.gray_img,k)
        img_for_corners = np.copy(self.colored_img)
        img=Harris.thresholding(threshold,harris_response,img_for_corners)
        Harris.Display(img,self.ui.output_img_2)
        
        
        
        
    
    
    
    
    