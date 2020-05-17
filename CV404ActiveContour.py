
import cv2 as cv2
import matplotlib.cm as cm
import numpy as np
import pylab as plb
import copy
from PyQt5.QtCore import QSize,pyqtSlot, QTimer,QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage
import os
from PyQt5 import QtCore, QtWidgets , QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget , QApplication,QPushButton,QLabel,QInputDialog,QSpinBox,QFileDialog,QProgressBar,QLineEdit,QMessageBox
import time
class ActiveContour:
    
    def apply_clicked(self):
        
        print(self.x)
        print(self.y)
        alpha=int(self.ui.textEdit.toPlainText())
        beta=int(self.ui.textEdit_2.toPlainText())
        gama=int(self.ui.textEdit_3.toPlainText())
        radius,ok=QInputDialog.getInt(self,"integer input dialog","enter min radius")
        #ActiveContour.activeContour(self.path ,(self.x, self.y), 30,300,2,80)
        ActiveContour.activeContour(self.path ,(self.x, self.y), radius,alpha,beta,gama)
        path="Activecontour.jpg"
        self.img=QPixmap(path)
        self.ui.Input_img_4.setPixmap(self.img)
        self.ui.Input_img_4.setScaledContents(True)
        
    def clear_clicked(self):    
        img=QPixmap(self.path)
        self.ui.Input_img_4.setPixmap(img)
        self.ui.Input_img_4.setScaledContents(True)
    
    def button_clicked(self):  
        
        
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter -- All Files (*);;Python Files (*.py)")
        self.path = fileName
        if fileName :
            

            self.pixmap = QPixmap(fileName)
            self._img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            print(self.pixmap.width())
            print(self.pixmap.height())
            self.ui.Input_img_4.setScaledContents(True)
            self.ui.Input_img_4.setPixmap(self.pixmap)
            self.ui.label_32.setText(os.path.basename(fileName))
            self.ui.label_33.setText(str(self.pixmap.height()) +"X"+str(self.pixmap.width()))
            time.sleep(.300)
            QMessageBox.warning(self,"Message","click on the center of object by the mouse")
            
    def internalEnergy(snake,Alpha = 300,Beta = 2,Beta_arr=None):
        iEnergy=0
        snakeLength=len(snake)
        if (Beta_arr==None):
           beta_arr=[Beta]*snakeLength
        else:
           beta_arr=Beta_arr
        for index in range(snakeLength-1,-1,-1):
            nextPoint = (index+1)%snakeLength
            currentPoint = index % snakeLength
            previousePoint = (index - 1) % snakeLength
            con_en= (Alpha *(np.linalg.norm(snake[nextPoint] - snake[currentPoint] )**2))
            curv_en=(beta_arr[index]*(np.linalg.norm(snake[nextPoint] - 2 * snake[currentPoint] + snake[previousePoint])**2))
            iEnergy = iEnergy+ con_en+curv_en
        return iEnergy
    # external forces summation of image grediant for all contour points
    
    def normalize_gradiant(image, from_min, from_max, to_min, to_max):
        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled = np.array((image - from_min) / float(from_range), dtype=float)
        return to_min + (scaled * to_range)
    
    def ImageGradiant(image):
        s_mask = 17
        sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=s_mask))
        sobelx = ActiveContour.normalize_gradiant(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
        sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=s_mask))
        sobely = ActiveContour.normalize_gradiant(sobely, np.min(sobely), np.max(sobely), 0, 255)
        gradient = 0.5 * sobelx + 0.5 * sobely
        return gradient
    
    def Gradient_fun(gradient, snak):
        sum = 0
        snaxels_Len= len(snak)
        for index in range(snaxels_Len-1):
            point = snak[index]
            sum = sum+((gradient[point[1]][point[0]]))
        return sum
    def externalEnergy(grediant,image,snak,gamma=80):
        _W_LINE = 80
        sum = 0
        snaxels_Len = len(snak)
        for index in range(snaxels_Len - 1):
            point = snak[index]
            sum = +(image[point[1]][point[0]])
        pixel = 255 * sum
    
        eEnergy = _W_LINE*pixel -gamma*ActiveContour.Gradient_fun(grediant, snak)
    
        return eEnergy
    # total energy for internal and external without constant
    def totalEnergy(grediant, image, snake,alpha,beta,gamma,beta_arr):
        iEnergy = ActiveContour.internalEnergy(snake,alpha,beta,beta_arr)
        eEnergy=ActiveContour.externalEnergy(grediant, image, snake,gamma)
        tEnergy = iEnergy+eEnergy
    
        return tEnergy
    def Beta_effect (snakes,beta,th,th_m,grediant):
        new_beta=[]
        beta_en=[]
        mag_gr=np.linalg.norm(grediant)
        snakeLength=len(snakes)
        for index in range(snakeLength-1,-1,-1):
            nextPoint = (index+1)%snakeLength
            currentPoint = index % snakeLength
            previousePoint = (index - 1) % snakeLength
            en=np.linalg.norm((snakes[currentPoint]/np.linalg.norm(snakes[currentPoint]))-(snakes[nextPoint]/np.linalg.norm(snakes[nextPoint])))**2
            beta_en.append(en)
        for index in range(snakeLength-1,-1,-1):
            nextPoint = (index+1)%snakeLength
            currentPoint = index % snakeLength
            previousePoint = (index - 1) % snakeLength
            if((beta_en[currentPoint]> beta_en[previousePoint]) and (beta_en[currentPoint]> beta_en[nextPoint]) and (beta_en[currentPoint]> th) and( mag_gr>th_m)):
              new_beta.append(0)
            else:
              new_beta.append(beta)
        return new_beta 
        
    #points
    def isPointInsideImage(image, point):
    
        return np.all(point < np.shape(image)) and np.all(point > 0)
    
    
    def _pointsOnCircle(center, radius, num_points=12):
        points = np.zeros((num_points, 2), dtype=np.int32)
        for i in range(num_points):
            theta = float(i)/num_points * (2 * np.pi)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            p = [x, y]
            points[i] = p
            
        return points
    
    #plot
    def display_snakes(image, changedPoint=None, snaxels=None):
    
        plb.clf()
        if snaxels is not None:
            for s in snaxels:
                if(changedPoint is not None and (s[0] == changedPoint[0] and s[1] == changedPoint[1])):
                    plb.plot(s[0], s[1], 'r', markersize=10.0)
                else:
                    plb.plot(s[0],s[1],'g.',markersize=10.0)
    
        plb.imshow(image, cmap=cm.Greys_r)
        plb.draw()
        
        return
    
    def activeContour(image_file, center, radius,alpha,beta,gamma):
        
        neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])
        ptsmoved= 0
        image = cv2.imread(image_file, 0)
        plb.ion()
        plb.figure(figsize=np.array(np.shape(image)) / 50.)
    
        snake = ActiveContour._pointsOnCircle(center, radius, 20)
        beta_arr=[beta]*len(snake)
        grediant = ActiveContour.ImageGradiant(image)
        plb.ioff()
    
        snakeColon =  copy.deepcopy(snake)
    
        for i in range(100):
            for index,point in enumerate(snake):
                min_energy2 = float("inf")
                for cindex,movement in enumerate(neighbors):
                    next_node = (point +movement)
                    if not ActiveContour.isPointInsideImage(image, next_node):
                        continue
                    if not ActiveContour.isPointInsideImage(image, point):
                        continue
    
                    snakeColon[index]=next_node
    
                    totalEnergyNext = ActiveContour.totalEnergy(grediant, image, snakeColon,alpha,beta,gamma,beta_arr)
                    if(totalEnergyNext > min_energy2):
                        ptsmoved=  ptsmoved+1
                    else:
                        min_energy2 = copy.deepcopy(totalEnergyNext)
                        indexOFlessEnergy = copy.deepcopy(cindex)
                snake[index] = (snake[index]+neighbors[indexOFlessEnergy])
                if(ptsmoved>0.2*ptsmoved):
                          beta_new=ActiveContour.Beta_effect (snake,beta,10,8,grediant)
                          beta_arr=beta_new
            snakeColon = copy.deepcopy(snake)
                
    
        plb.ioff()
        ActiveContour.display_snakes(image,None, snake)
        plb.plot()
        plb.savefig('Activecontour.jpg')
        plb.axis('off')
        plb.savefig("Activecontour.jpg",bbox_inches='tight',pad_inches = 0)
        plb.show()
    
        return

#def _test():
#
#    activeContour("brainTumor.png", (98, 152), 30,300,2,80)
#    return
#
#if __name__ == '__main__':
#    _test()