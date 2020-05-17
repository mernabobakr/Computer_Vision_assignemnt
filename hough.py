import numpy as np
import matplotlib.pyplot as plt
import cv2

# read in shapes image and convert to grayscale
shapes = cv2.imread('b.jpg')
cv2.imshow('Original Image', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
###########################################
# Step 1
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) 
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1 #pixel voted up
    return H, rhos, thetas

def hough_lines_peaks(H, num_peaks, threshold=0, nhood_size=3):
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) 
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)
        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx 
       ###
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        ###
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x),int( max_x)):
            for y in range(int(min_y), int(max_y)):
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255
    return indicies, H

# drawing the lines from the Hough Accumulatorlines
def hough_lines_draw(img, indicies, rhos, thetas):
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def hough_lines_detection(image,num_peaks=20,nhood_size=7):
        shapes_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
        canny_edges = cv2.Canny(shapes_blurred, 70, 150)
        H, rhos, thetas = hough_lines_acc(canny_edges)
        indicies, H = hough_lines_peaks(H,num_peaks,10,nhood_size) # find peaks
        hough_lines_draw(image, indicies, rhos, thetas)
        
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

def hough_circles_draw(A):
    fig = plt.figure()
    plt.imshow(shapes)
    circleCoordinates = np.argwhere(A)
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()

def hough_circles_detection(image):
    shapes_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
    canny_edges = cv2.Canny(shapes_blurred, 100, 200)    
    res = hough_circles(canny_edges,8.1,15,radius=[50,5])
    hough_circles_draw(res)
    
          
        
x=0
shapes_grayscale = cv2.cvtColor(shapes,cv2.COLOR_RGB2GRAY)
shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
canny_edges = cv2.Canny(shapes_blurred, 70, 150)
cv2.imshow('canny', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
#hough_circles_detection(shapes)
cv2.imshow('Major Lines: Manual Hough Transform', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
        
        
        
        
        
