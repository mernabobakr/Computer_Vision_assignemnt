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
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from gui import Ui_MainWindow
class Histogram:
    
        
        
        
    def load_image(self):  
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            self.colored_img = io.imread(self.path)
            self.gray_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.image1 = QPixmap(fileName)
            Histogram.Display(self.gray_img,self.ui.label_histograms_input)
            Histogram.Display(self.colored_img,self.ui.label_histograms_hinput_3)
          
            
            self.ui.label_11.setText(os.path.basename(fileName))
            self.ui.label_10.setText(str(self.image1.height()) +"X"+str(self.image1.width()))    
        
        
          
    def Display( img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image)    
            
   
        
    def Plot_colored_Histogram(Histogram):
    	plt.figure()
    	plt.title("Color Image Histogram")
    	plt.xlabel("Intensity Level")
    	plt.ylabel("Intensity Frequency")
    	plt.xlim([0, 256])
    	plt.plot(Histogram[:,0],'b') 
    	plt.plot(Histogram[:,1],'g') 
    	plt.plot(Histogram[:,2],'r') 
    	plt.savefig("Color_Histogram.jpg")    
        
    



    
    
    def Histogram_colored(Imagee):
	
    	Image_Height = Imagee.shape[0]
    	Image_Width = Imagee.shape[1]
    	Image_Channels = Imagee.shape[2]
    	
    	Histogram = np.zeros([256, Image_Channels], np.int32)
    	
    	for x in range(0, Image_Height):
    		for y in range(0, Image_Width):
    			for c in range(0, Image_Channels):
    					Histogram[Imagee[x,y,c], c] +=1
    	
    	return Histogram 
    
    
    
    def colored_histogram_button_pressed(self):
        histogram = Histogram.Histogram_colored(self.colored_img)
        Histogram.Plot_colored_Histogram(histogram)
        path="Color_Histogram.jpg"
        img=QPixmap(path)
        self.ui.label_histograms_hinput_4.setPixmap(img)
        self.ui.label_histograms_hinput_4.setScaledContents(True)
    
    def get_histogram(image, bins):
    
        histogram = np.zeros(bins)
        
        
        for pixel in image:
            histogram[pixel] += 1
        
       
        return histogram    
    
    def cumsum(a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)
    
    def normalize(cs):
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()
        cs_normal = nj / N
        cs_integer= cs_normal.astype('uint8')
        return cs_integer 
    def input_histogram(self):
        self.img_as_array = np.asarray(self.gray_img)
        self.flat = self.img_as_array.flatten()
        self.input_histo = Histogram.get_histogram(self.flat, 256)
        plt.figure()
        plt.plot(self.input_histo)
        plt.savefig("input_Histogram.jpg")
        path="input_Histogram.jpg"
        img=QPixmap(path)
        self.ui.label_histograms_hinput.setPixmap(img)
        self.ui.label_histograms_hinput.setScaledContents(True)
        Histogram.output_histogram(self)
    def output_histogram(self):
        cs = Histogram.cumsum(self.input_histo)
        integer_histo=Histogram.normalize(cs)
        self.img_new = integer_histo[self.flat]
        plt.figure()
        output_histo = Histogram.get_histogram(self.img_new, 256)
        plt.plot(output_histo)
        plt.savefig("output_Histogram.jpg")
        path="output_Histogram.jpg"
        img=QPixmap(path)
        self.ui.label_histograms_houtput.setPixmap(img)
        self.ui.label_histograms_houtput.setScaledContents(True)
        Histogram.get_image_after_equalization(self)
    def get_image_after_equalization(self):
        arr = np.reshape(self.img_new, self.img_as_array.shape)
        #image=Image.fromarray(arr)
        Histogram.Display(arr,self.ui.label_histograms_output)

    


