
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
from PyQt5.QtGui import QPixmap,QPen,QPainter,QBrush,QColor
from collections import defaultdict

class Hough:
    def load_image(self):  
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            self.colored_img = cv2.imread(self.path)
            self.gray_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.image1 = QPixmap(fileName)
            Hough.Display(self.gray_img,self.ui.Input_img_2)
            self.ui.label_27.setText(os.path.basename(fileName))
            self.ui.label_28.setText(str(self.image1.height()) +"X"+str(self.image1.width()))  
            




        
    def Display( img, label):
        yourQImage = qimage2ndarray.array2qimage(img)
        pixmap = QPixmap(QPixmap.fromImage(yourQImage))
        image = pixmap.scaled(pixmap.width(), pixmap.height())
        label.setScaledContents(True)
        label.setPixmap(image)     
        
    def hough_line(image):
    
        Ny = image.shape[0]
        Nx = image.shape[1]
        Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
        thetas = np.deg2rad(np.arange(0, 180))
        rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
        accumulator = np.zeros((2 * Maxdist, len(thetas)))
        for y in range(Ny):
            for x in range(Nx):
                 if image[y,x] > 0:
                     for k in range(len(thetas)):
                    
                         r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                         accumulator[int(r) + Maxdist,k] += 1
        return accumulator, thetas, rs
    
    def extract_lines(accumulator,thetas,rhos,threshold):
                      lines = defaultdict()
                      acc2 = np.zeros(accumulator.shape)
                      for rho_idx in range(len(rhos)) :
                        for theta_idx in range(len(thetas)) :
                            if accumulator[rho_idx, theta_idx] > threshold :
                                theta = thetas[theta_idx]
                                rho = rhos[rho_idx]
                                lines[(rho,theta)] = accumulator[rho_idx, theta_idx]
                                
                                acc2[rho_idx,theta_idx] = accumulator[rho_idx, theta_idx]
                      return lines,acc2
                  
                    
    def draw_line(img,accumulator,thetas,rhos):
        lines,acc2 = Hough.extract_lines(accumulator,thetas,rhos,80)
        for (rho,theta), val in lines.items():
            a = np.cos(theta)
            b = np.sin(theta)
            pt0 = rho*np.array([a,b])
            # these are then scaled so that the lines go off the edges of the image
            pt1 = tuple((pt0 + 1000* np.array([-b,a])).astype(int))
            pt2 = tuple((pt0 - 1000* np.array([-b,a])).astype(int))
            cv2.line(img, pt1, pt2, (0,255, 0), 3)
            
        
            
            
            
            
            
                
        def hough_circles(img,threshold,region,radius = None):
            (M,N) = img.shape
            if radius == None:
                R_max = np.max((M,N))
                R_min = 3
            else:
                [R_max,R_min] = radius
        
            R = R_max - R_min
            #Initializing accumulator array.
            A = np.zeros((R_max,M+2*R_max,N+2*R_max))
            B = np.zeros((R_max,M+2*R_max,N+2*R_max))
            theta = np.arange(0,360)*np.pi/180
            edges = np.argwhere(img[:,:])
            for val in range(R):
                r = R_min+val
                #Creating a Circle Blueprint
                bprint = np.zeros((2*(r+1),2*(r+1)))
                (m,n) = (r+1,r+1)                                                       
                for angle in theta:
                    x = int(np.round(r*np.cos(angle)))
                    y = int(np.round(r*np.sin(angle)))
                    bprint[m+x,n+y] = 1
                constant = np.argwhere(bprint).shape[0]
                for x,y in edges:                                                       
                    X = [x-m+R_max,x+m+R_max]                                           
                    Y= [y-n+R_max,y+n+R_max]                                            
                    A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
                A[r][A[r]<threshold*constant/r] = 0
        
            for r,x,y in np.argwhere(A):
                temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
                try:
                    p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
                except:
                    continue
                B[r+(p-region),x+(a-region),y+(b-region)] = 1
        
            return B[:,R_max:-R_max,R_max:-R_max]
    
    def hough_circles_draw(A,shapes):
        fig = plt.figure()
        plt.imshow(shapes)
        circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
        circle = []
        for r,x,y in circleCoordinates:
            circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
        
       
        
        plt.axis('off')
        plt.savefig("detectedcircles.jpg",bbox_inches='tight',pad_inches = 0)  
        
    
    def hough_circles_detection(image,threshold,maxx,minn):
        shapes_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
        canny_edges = cv2.Canny(shapes_blurred, 100, 200)    
        res = Hough.hough_circles(canny_edges,threshold,15,radius=[maxx,minn])
        Hough.hough_circles_draw(res,image)
        
        
    
#    def frame(self,event):
#        self.x=math.floor((event.pos().x()*self.size)/self.ui.output_img_2.frameGeometry().width())
#        print (self.x)
#        self.y=math.floor((event.pos().y()*self.size)/self.ui.output_img_2.frameGeometry().height())
#        print(self.y)
#        QApplication.processEvents()
#        self.painterInstance = QPainter(self.img )   #b3mel opject
#        self.painterInstance.begin(self)  
#        self.penRectangle =QPen(QtCore.Qt.red)  #yehdd el elon
#        self.penRectangle.setWidth(1)
#        self.penPoint =QPen(QtCore.Qt.blue)
#        self.penPoint.setWidth(1)  #
#        self.painterInstance.setPen(self.penPoint)  #apply el lon
#        self.painterInstance.drawRect(self.x,self.y,1,1)
#        self.painterInstance.setPen(self.penRectangle)
#        self.painterInstance.drawRect(self.x-5,self.y-5,10,10)
#        self.painterInstance.end()
#        result=self.colored_img .scaled(int(self.ui.output_img_2.height()), int(self.ui.output_img_2.width()),QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) #scale 3la elabel
#        self.ui.output_img_2.setPixmap(result)
#        self.painterInstance.end()
#        QApplication.processEvents()
#    
    
        
    def apply_button_clicked(self,event):
        
        if self.ui.comboBox_7.currentIndex()==1:
            #img=Hough.hough_lines_detection(self.colored_img,num_peaks=20,threshold=th,nhood_size=7)'
            cannyImg = cv2.Canny(self.colored_img,100,200)
            accumulator,thetas,rhos = Hough.hough_line(cannyImg)

            Hough.draw_line(self.colored_img,accumulator,thetas,rhos)
            cv2.imwrite('Lines.jpg',self.colored_img)
            path="Lines.jpg"
            self.img=QPixmap(path)
            self.ui.output_img.setPixmap(self.img)
            self.ui.output_img.setScaledContents(True)
            
        elif self.ui.comboBox_7.currentIndex()==0:
            th,ok=QInputDialog.getInt(self,"integer input dialog","enter threshold")
            mini,ok=QInputDialog.getInt(self,"integer input dialog","enter min radius")
            maxi,ok=QInputDialog.getInt(self,"integer input dialog","enter maxi radius")
            Hough.hough_circles_detection(self.colored_img,th,maxi,mini)
            path="detectedcircles.jpg"
            self.img=QPixmap(path)
            self.ui.output_img.setPixmap(self.img)
            self.ui.output_img.setScaledContents(True)
            Hough.frame(self,event)
        