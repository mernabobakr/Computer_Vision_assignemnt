from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.feature import corner_harris, corner_peaks
from scipy import signal as sig
import numpy as np
import cv2

img = cv2.imread('images/House.jpg')
imggray = rgb2gray(img)

#plt.imshow(imggray, cmap="gray")
#plt.axis("off")
#plt.show()

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')




def get_harris_response(imggray,k):
    I_x = gradient_x(imggray)
    I_y = gradient_y(imggray)
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
    
    harris_response = detA - k * traceA ** 2
    return harris_response 

harris_response=get_harris_response(imggray,0.05)

 
         
img_for_corners = np.copy(img)
#img_copy_for_edges = np.copy(img)
def thresholding(threshold,harris_res,img_copy_for_corners):        
    
    for rowindex, point in enumerate(harris_res):
        for colindex, r in enumerate(point):
            if np.abs(r) > threshold * np.max(r):
                # this is a corner
                img_copy_for_corners[rowindex, colindex] = [255,0,0]

    return img_copy_for_corners

img_copy_for_corners=thresholding(2,harris_response,img_for_corners)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax[0].set_title("corners found")

ax[0].imshow(img_copy_for_corners)

plt.show()        