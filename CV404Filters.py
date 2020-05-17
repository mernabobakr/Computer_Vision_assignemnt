
from PyQt5 import QtCore, QtWidgets , QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
from PyQt5.QtCore import QSize,pyqtSlot, QTimer,QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage
import numpy as np
import sys
import os
from scipy import signal  

from gui import Ui_MainWindow
import cv2
import qimage2ndarray


class filters:
   
       


    def button_clicked(self):  
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            

            self.pixmap = QPixmap(fileName)
            self._img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            print(self.pixmap.height())
            self.ui.label_filters_input.setScaledContents(True)
            self.ui.label_filters_input.setPixmap(self.pixmap)
            self.ui.label.setText(os.path.basename(fileName))
            self.ui.label_2.setText(str(self.pixmap.height()) +"X"+str(self.pixmap.width()))
          
    def noise_combobox_changed(self):
         self.img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
         if self.ui.comboBox_2.currentIndex()==0:
            print("Gaussian")
            
            self.img=filters.im_gaussian_noise(3,10,self.img)
            filters.Display(self.img,self.ui.label_filters_input)
            
         elif self.ui.comboBox_2.currentIndex()==1:
             print("salt&pepper")
             
             self.img=filters.salt_pepper_noise(self.img,0.2)
             filters.Display(self.img,self.ui.label_filters_input)
         elif self.ui.comboBox_2.currentIndex()==2:
             print("uniform")
             
             self.img=filters.im_uniform_noise(2,200,self.img)
             filters.Display(self.img,self.ui.label_filters_input)
    def filter_combobox_changed(self):
         img=self.img
         if self.ui.comboBox.currentIndex()==0:
            print("avg filter")
            #img1 = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            img=filters.avg_filter(img)
            filters.Display(img,self.ui.label_filters_output)
         elif self.ui.comboBox.currentIndex()==1:
             print("median")
             img=filters.median_filter(3,img)
             filters.Display(img,self.ui.label_filters_output)
         elif self.ui.comboBox.currentIndex()==2:
             img=filters.gaussian_filter(img)
             filters.Display(img,self.ui.label_filters_output)
             print("gaussian")  
       
    
    def edge_combobox_changed(self):
        img=self._img 
        if self.ui.comboBox_3.currentIndex()==0:
            detected_img=filters.Roberts_detection(img)
            filters.Display(detected_img,self.ui.label_filters_output)
        elif self.ui.comboBox_3.currentIndex()==1:
            detected_img=filters.canny_detection(img)
            filters.Display(detected_img,self.ui.label_filters_output)
            
        elif self.ui.comboBox_3.currentIndex()==2:
            detected_img=filters.sobel_detection(img)
            filters.Display(detected_img,self.ui.label_filters_output)
    
    
        elif self.ui.comboBox_3.currentIndex()==3:
            detected_img=filters.prewitt_detection(img)
            filters.Display(detected_img,self.ui.label_filters_output)
    def gaussian_filter(img):
        averageMask=filters.gaussion_filter(3)
        filteredImg=signal.convolve2d(img,averageMask)
        return filteredImg 
         
    def avg_filter(img):
        averageMask=filters.average_filter_mask(3,3)
        filteredImg=signal.convolve2d(img,averageMask)
        return filteredImg


        
    def corr(img,mask):
        row,col=img.shape
        m,n=mask.shape
        new_img=np.zeros((row+m-1,col+n-1))
        filtered_img=np.zeros(img.shape)
        m=m//2
        n=n//2 
        new_img[m:new_img.shape[0]-m,n:new_img.shape[1]-n]=img
        for i in range(m,new_img.shape[0]-m):
            for j in range(n,new_img.shape[1]-n):
                temp=new_img[i-m:i+m+1,j-n:j+n+1]
                result=temp*mask
                filtered_img[i-m,j-n]=result.sum()
        return  filtered_img
    #filters
    def average_filter_mask(m,n):
        average=np.ones((m,n))/m*n
        return average

    
    def gaussion_filter(size):
        
        sigma=1
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    def median_filter(m,img): #m:mask_size
          m=m//2   
          img_median = img.copy()
          height = img.shape[0]
          width = img.shape[1] 
          for i in np.arange(m, height-m):
                for j in np.arange(m, width-m):
                    neighbors = []
                    for k in np.arange(-m, m+1):
                        for l in np.arange(-m,m+1):
                            a = img.item(i+k, j+l)
                            neighbors.append(a)
                    neighbors.sort()
                    median = neighbors[len(neighbors)//2]
                    b = median
                    img_median.itemset((i,j), b)
          return  img_median



     #noise
    def gaussian_noise( mu, sigma, im_size ):
        randGaussian=np.random.normal( mu, sigma, im_size) #np.random.normal Gaussian noise
        return randGaussian
    def im_gaussian_noise(mu, sigma, im):
        g_noise= filters.gaussian_noise(mu,sigma, im.shape)
        img_w_g_noise = im + g_noise
        return img_w_g_noise
    def salt_pepper_noise(img,percent):
            img_noisy=np.zeros(img.shape)
            salt_pepper = np.random.random(img.shape) # Uniform distribution
            cleanPixels_ind=salt_pepper > percent
            #NoisePixels_ind=salt_pepper <= percent
            pepper = (salt_pepper <= (0.5* percent)); # pepper < half percent
            salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent))
            img_noisy[cleanPixels_ind]=img[cleanPixels_ind]
            img_noisy[pepper] = 0
            img_noisy[salt] = 1
            return img_noisy    
        
    def Uniform_noise(Low, high, im_size ):
        Uniform=np.random.uniform(Low, high, im_size) #np.random.normal Gaussian noise
        return Uniform
    def im_uniform_noise(low,high, im):
        uni_noise= filters.Uniform_noise( low, high, im.shape)
        img_w_ui_noise = im + uni_noise
        return img_w_ui_noise

   #Edge detection
        
    def sobel_detection(img):
          Hx =  np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
          Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
          Gx=signal.convolve2d(img,Hx)
          Gy=signal.convolve2d(img,Hy)
          img_out_sobel=Gx+Gy
          return img_out_sobel
    
    def prewitt_detection(img):
          Hx = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

          Hy = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
          Gx=signal.convolve2d(img,Hx)
          Gy=signal.convolve2d(img,Hy)
          img_out_prewitt =np.hypot(Gx,Gy)
          img_out_prewitt= ( img_out_prewitt/np.max( img_out_prewitt))*255
          return img_out_prewitt


    def Roberts_detection(img):
          Hx = np.array( [[1, 0 ],
                          [ 0,-1]])

          Hy = np.array( [[0, 1],
                          [-1, 0]])
          Gx=signal.convolve2d(img,Hx)
          Gy=signal.convolve2d(img,Hy)
          img_out_Roberts =np.hypot(Gx,Gy)
          img_out_Roberts= ( img_out_Roberts/np.max(img_out_Roberts))*255
          return img_out_Roberts
        
    def Canny_detection_magnitude_direction(img):
        Hx =  np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
                             
        Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        img_x=signal.convolve2d(img,Hx)
        img_y=signal.convolve2d(img,Hy)
        gradient_magnitude =np.hypot(img_x,img_y)
        gradient_magnitude= (gradient_magnitude/np.max(gradient_magnitude))*255
        gradient_direction = np.arctan2(img_y,img_x)
        return  gradient_magnitude,gradient_direction
    #step3:Aplly Non_max suppression:
    def non_max_suppression(img,gradient_direction):
        row,col = img.shape
        Z = np.zeros((row,col), dtype=np.int32)
        angle = gradient_direction * 180. / np.pi
        angle[angle < 0] += 180
    
        
        for i in range(1,row-1):
            for j in range(1,col-1):
                    before_pixel = 255
                    after_pixel = 255
                    
                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        before_pixel = img[i, j+1]
                        after_pixel = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        before_pixel = img[i+1, j-1]
                        after_pixel = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                       before_pixel = img[i+1, j]
                       after_pixel = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        before_pixel = img[i-1, j-1]
                        after_pixel = img[i+1, j+1]
    
                    if (img[i,j] >= before_pixel) and (img[i,j] >= after_pixel):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0
        return Z
    #step4:apply threshold
    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = np.max(img)* highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        M, N = img.shape
        out_img = np.zeros((M,N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        out_img[strong_i, strong_j] = strong
        out_img[weak_i, weak_j] = weak
        
        return( out_img, weak, strong)
    #step5:apply hysteresis
    def hysteresis(img, weak, strong=255):
        M, N = img.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
        return img

    #canny_detection 
    def canny_detection(img):
      gaussion_filter_mask=filters.gaussion_filter(3)
      smooth_image=signal.convolve2d(img,gaussion_filter_mask)
      magnitude,direction =filters.Canny_detection_magnitude_direction(smooth_image)
      non_max_supression=filters.non_max_suppression(magnitude,direction)
      res, weak, strong=filters.threshold(non_max_supression)
      hysteresis_1=filters.hysteresis(res,weak)
      return hysteresis_1
      
      
    
      
      
        
                      
                    
            

        
        
    
       
      
      


        
    def Display(img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image)    
        
            
