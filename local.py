import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import skimage.io
import skimage.viewer
import skimage.color
from skimage import color
import cv2
import matplotlib.pyplot as plt

def main():
    
   
    img=skimage.io.imread('images/some-pigeon.jpg')
    gray_img=color.rgb2gray(img)
    
    block_size = 513
    constant = 2
    th1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    th2 = cv2.adaptiveThreshold (img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    
    output = [img, th1, th2]
    
    titles = ['Original', 'Mean Adaptive', 'Gaussian Adaptive']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(output[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()  

if __name__ == "__main__":
    main()