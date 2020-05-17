import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import skimage.io
import skimage.viewer
import skimage.color
from skimage import color
import cv2
img = cv2.imread('images/color.png', cv2.IMREAD_GRAYSCALE)

image = io.imread('images/girlWithScarf.png')
img_as_array = np.asarray(img)


#### thresholding#########
def thresholding(image,threshold):
    gray_img=color.rgb2gray(image)
    img_as_array = np.asarray(gray_img)
    height = img_as_array.shape[0]
    width = img_as_array.shape[1]
    for i in np.arange(height):
        for j in np.arange(width):
            a = img_as_array.item(i,j)
            if a > threshold:
                b = 255
            else:
                b = 0
            img_as_array.itemset((i,j), b)
    return img_as_array 

###############################

#histogram and equalization
flat = img_as_array.flatten()
def get_histogram(image, bins):
    
    histogram = np.zeros(bins)
    
    
    for pixel in image:
        histogram[pixel] += 1
    
   
    return histogram

 
#cumulative function
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)
#normalization 
def normalize(cs):
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs_normal = nj / N
    cs_integer= cs_normal.astype('uint8')
    return cs_integer 



#to plot histogram og input image
input_histo = get_histogram(flat, 256)
plt.plot(input_histo)
##to start equalization and get new image 
cs = cumsum(input_histo)
integer_histo=normalize(cs)
img_new = integer_histo[flat]
##to get new histogram
plt.figure()
output_histo = get_histogram(img_new, 256)
plt.plot(output_histo)
##to show output image after equalization
output_image = np.reshape(img_new, img_as_array.shape)
##to dispplay input and output image (before and after equalization )
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.add_subplot(1,2,1)
plt.imshow(image, cmap='gray')
fig.add_subplot(1,2,2)
#plt.imshow(output_image, cmap='gray')
plt.show(block=True)







######### to plot thresholded image ####
#plt.figure()
#thresholded_img=thresholding(image,0.5)
#plt.imshow(thresholded_img, cmap='gray') 
#