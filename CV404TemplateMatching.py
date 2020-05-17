# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:58 2020

@author: NADA
"""

import qimage2ndarray
from PyQt5 import QtCore, QtWidgets , QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
from PyQt5.QtCore import QSize,pyqtSlot,QTimer,QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from gui import Ui_MainWindow
import os
import numpy as np
from scipy.signal import correlate2d
import itk
from itkwidgets import view
import itkwidgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
import cv2
import nms
import importlib
import time
import scipy.ndimage.filters as filters
importlib.reload(nms)


class TM:
    def Display( img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image)    
            
    
    def load_first_image(self):  
       
        print("zz")
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path1 = fileName
        if fileName :
            
            img=Image.open(self.path1)
            img_rgb=np.array(img)
            self.img_gray=TM.rgb2gray(img_rgb)
            self.image1 = QPixmap(fileName)
            self.gray_img = cv2.imread(self.path1, cv2.IMREAD_GRAYSCALE)
            TM.Display(self.gray_img,self.ui.Template_matching_inputA)
            
            
            self.ui.label_54.setText(os.path.basename(fileName))
            self.ui.label_53.setText(str(self.image1.height()) +"X"+str(self.image1.width()))
            
            
    def load_second_image(self): 
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path2 = fileName
        if fileName :
            
            Temp=Image.open(self.path2)
            Temp_rgb=np.array(Temp)
            self.Temp_gray=TM.rgb2gray(Temp_rgb)
            self.image2 = QPixmap(fileName)
            self.gray_temp = cv2.imread(self.path2, cv2.IMREAD_GRAYSCALE)
            TM.Display(self.gray_temp,self.ui.Template_matching_inputB)
            
            self.ui.label_56.setText(os.path.basename(fileName))
            self.ui.label_55.setText(str(self.image2.height()) +"X"+str(self.image2.width()))
    
    
    def match_template_corr( x , temp ):
        y = np.empty(x.shape)
        y = correlate2d(x,temp,'same')
        return y
    
    
    def match_template_corr_zmean( x , temp ):
        return TM.match_template_corr(x , temp - temp.mean())
    
    
    def match_template_ssd( x , temp ):
        term1 = np.sum( np.square( temp ))
        term2 = -2*correlate2d(x, temp,'same')
        term3 = correlate2d( np.square( x ), np.ones(temp.shape),'same' )
        ssd = np.maximum( term1 + term2 + term3 , 0 )
        return 1 - np.sqrt(ssd)
    
    def match_template_xcorr( f , t ):
        f_c = f - correlate2d( f , np.ones(t.shape)/np.prod(t.shape), 'same') 
        t_c = t - t.mean()
        numerator = correlate2d( f_c , t_c , 'same' )
        d1 = correlate2d( np.square(f_c) , np.ones(t.shape), 'same')
        d2 = np.sum( np.square( t_c ))
        denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
        response = np.zeros( f.shape )
        valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
        response[valid] = numerator[valid]/denumerator[valid]
        return response
    
    def rgb2gray(img):
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    
    
    
    def get_maxima_space(temp,space):
        
        #Get the local maxima
        htemp, wtemp = temp.shape
        data_max = filters.maximum_filter(space,np.minimum(htemp,wtemp))
        maxima_space = np.copy(space)
        maxima_space[np.where(space!=data_max)] = 0
    
        return maxima_space
    
    
    
    
    
    
    
    
    def draw_detected_objects(img,temp,space,maxima_space,similarity_ratio=0.6):
        
        threshould = maxima_space.max()*similarity_ratio
        y_pos, x_pos = np.where(maxima_space>=threshould)
        htemp, wtemp = temp.shape
        
        #convert image to 3d image to draw on
        if (len(img.shape)<3):
           new_img = np.zeros([img.shape[0],img.shape[1],3],np.uint8)
           new_img[:,:,0] = img
           new_img[:,:,1] = img
           new_img[:,:,2] = img
        else:
           new_img = np.copy(img)
           
        #convert space to 3d image to draw on
        minimum = space.min()
        if minimum <0:
            space = space - minimum
            minimum = 0
        maximum = space.max()
        space = ((space-minimum)/(maximum-minimum))*255
        space = np.round(space)
        
        if (len(space.shape)<3):
           new_space = np.zeros([img.shape[0],img.shape[1],3],np.uint8)
           new_space[:,:,0] = space
           new_space[:,:,1] = space
           new_space[:,:,2] = space
        else:
           new_space = np.copy(space)
        
        #drawing the results
        for i in range(len(x_pos)):
            #draw circles around maxima in the space
            cv2.circle(new_space, (x_pos[i],y_pos[i]), 10, (0, 0, 255) , 2)
            #draw Rectangles on the main image
            cv2.rectangle(new_img, (int(x_pos[i]-wtemp/2), int(y_pos[i]-htemp/2)), (int(x_pos[i]+wtemp/2), int(y_pos[i]+htemp/2)), (0,0,255), 2)
    
        #draw maximums with different colors
        ymax,xmax = np.where(maxima_space==maxima_space.max())
        cv2.circle(new_space, (xmax[0],ymax[0]), 10, (0,255,0) , 2)
        cv2.rectangle(new_img, (int(xmax[0]-wtemp/2), int(ymax[0]-htemp/2)), (int(xmax[0]+wtemp/2), int(ymax[0]+htemp/2)), (0,255,0), 2)
        
        return new_img,new_space
    
    
    
    
    
    
    
    
    
    def image_to_qpixmap(img):
        imgg =np.copy(img)
        imgg = cv2.resize(imgg,(256,256))
        cv2.imwrite('imgg.jpg',imgg)
        pixmap = QPixmap('imgg.jpg')
        os.remove("imgg.jpg")        
        return pixmap
    def matching_pressed(self):
        if (self.ui.comboBox_4.currentIndex()==0):
            space = TM.match_template_corr(self.img_gray,self.Temp_gray)
        elif(self.ui.comboBox_4.currentIndex()==1):
            space =  TM.match_template_corr_zmean(self.img_gray,self.Temp_gray)
            
        elif(self.ui.comboBox_4.currentIndex()==2):
            space =  TM.match_template_ssd(self.img_gray,self.Temp_gray)
        elif(self.ui.comboBox_4.currentIndex()==3):   
            space =  TM.match_template_xcorr(self.img_gray,self.Temp_gray)
        self.start_record = time.time()    
        maximum_space=TM.get_maxima_space(self.Temp_gray,space)
        
        final_img,final_space= TM.draw_detected_objects(self.img_gray,self.Temp_gray,space,maximum_space)
        self.end_record = time.time()    
        self.time_tot=round(self.end_record-self.start_record)
        pixmap = TM.image_to_qpixmap(final_space)
        self.ui.Matchingspace.setPixmap(pixmap)
        self.ui.label_58.setText("The time elapsed is "+str(self.time_tot)+" seconds")
    
        pixmap = TM.image_to_qpixmap(final_img)
        self.ui.Detected_patterns.setPixmap(pixmap)    
    
    













