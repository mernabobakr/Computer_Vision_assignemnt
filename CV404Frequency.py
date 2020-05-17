from PyQt5 import QtCore, QtWidgets , QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
from PyQt5.QtCore import QSize,pyqtSlot,QTimer,QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import qimage2ndarray
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage 
import math
import sys
import os
import cv2
from gui import Ui_MainWindow

class Filter:

    def load_first_image(self):  
       
        print("zz")
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            
            self.img1 = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.image1 = QPixmap(fileName)
            
            self.ui.label_histograms_input_2.setScaledContents(True)
            self.ui.label_histograms_input_2.setPixmap(self.image1)
            self.ui.label_12.setText(os.path.basename(fileName))
            self.ui.label_13.setText(str(self.image1.height()) +"X"+str(self.image1.width()))
            
            
    def load_second_image(self): 
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            
    
            self.image2 = QPixmap(fileName)
            self.img2 = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.ui.label_histograms_hinput_2.setScaledContents(True)
            self.ui.label_histograms_hinput_2.setPixmap(self.image2)
            self.ui.label_15.setText(os.path.basename(fileName))
            self.ui.label_14.setText(str(self.image2.height()) +"X"+str(self.image2.width()))
            
          
       
    def make_hybrid(self):
        hyprid=Filter.hybridImage(self.img1,self.img2,25,10)
        Filter.Display(hyprid,self.ui.label_histograms_output_2)
    
    
    def scaleSpectrum(self,A):
       return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))
    
      
    
    
    def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
       centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
       centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
    
       def gaussian(i,j):
          coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
          return 1 - coefficient if highPass else coefficient
    
       return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])
    
    def Display(img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image)
    
    
    
    
    
    def filterDFT(imageMatrix, filterMatrix):
       shiftedDFT = fftshift(fft2(imageMatrix))
       
    
       filteredDFT = shiftedDFT * filterMatrix
       
       return ifft2(ifftshift(filteredDFT))
    
    
    
    
    def lowPass(imageMatrix, sigma):
       n,m = imageMatrix.shape
       return Filter.filterDFT(imageMatrix, Filter.makeGaussianFilter(n, m, sigma, highPass=False))
    
    
    def highPass(imageMatrix, sigma):
       n,m = imageMatrix.shape
       return Filter.filterDFT(imageMatrix, Filter.makeGaussianFilter(n, m, sigma, highPass=True))
    
    
    def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
       highPassed = Filter.highPass(highFreqImg, sigmaHigh)
       lowPassed = Filter.lowPass(lowFreqImg, sigmaLow)
    
       return highPassed + lowPassed
       
       
    




