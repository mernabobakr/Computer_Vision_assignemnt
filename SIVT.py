from scipy.signal import convolve2d
from cvutils_ass3 import gaussian_kernel2d
from skimage.transform import rescale
from math import sqrt,sin, cos
import logging
import sys
import numpy as np
import cv2
import importlib
import cvutils_ass3
importlib.reload(cvutils_ass3)
import matplotlib.pyplot as plt
from cvutils_ass3 import padded_slice, sift_gradient
from PIL import Image
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Test
logger = logging.getLogger('SIFT')

# The following are suggested by SIFT author
N_OCTAVES = 4 
N_SCALES = 5 
SIGMA = 1.6
K = sqrt(2)
SIGMA_SEQ = lambda s: [ (K**i)*s for i in range(N_SCALES) ] # (s, √2s , 2s, 2√2 s , 4s )
SIGMA_SIFT = SIGMA_SEQ(SIGMA) #
print(SIGMA_SEQ)
print(SIGMA_SIFT)
KERNEL_RADIUS = lambda s : 2 * int(round(s))
KERNELS_SIFT = [ gaussian_kernel2d(std = s, 
                                   kernlen = 2 * KERNEL_RADIUS(s) + 1) 
                for s in SIGMA_SIFT ]


def image_dog( img ):
    octaves = []
    dog = []
    base = rescale( img, 2, anti_aliasing=False) 
    octaves.append([ convolve2d( base , kernel , 'same', 'symm') 
                    for kernel in KERNELS_SIFT ])
    dog.append([ s2 - s1 
                for (s1,s2) in zip( octaves[0][:-1], octaves[0][1:])])
    for i in range(1,N_OCTAVES):
        base = octaves[i-1][2][::2,::2] # 2x subsampling 
        octaves.append([base] + [convolve2d( base , kernel , 'same', 'symm') 
                                 for kernel in KERNELS_SIFT[1:] ])
        dog.append([ s2 - s1 
                    for (s1,s2) in zip( octaves[i][:-1], octaves[i][1:])])
        logger.info('Done {}/{} octaves'.format(i+1, N_OCTAVES))
    return dog , octaves






#ratiobetween coords
def corners( dog , r = 10 ):
    threshold = ((r + 1.0)**2)/r
    dx = np.array([-1,1]).reshape((1,2))
    dy = dx.T
    dog_x = convolve2d( dog , dx , boundary='symm', mode='same' )
    dog_y = convolve2d( dog , dy , boundary='symm', mode='same' )
    dog_xx = convolve2d( dog_x , dx , boundary='symm', mode='same' )
    dog_yy = convolve2d( dog_y , dy , boundary='symm', mode='same' )
    dog_xy = convolve2d( dog_x , dy , boundary='symm', mode='same' )
    
    tr = dog_xx + dog_yy
    det = dog_xx * dog_yy - dog_xy ** 2
    response = ( tr**2 +10e-8) / (det+10e-8)
    
    coords = list(map( tuple , np.argwhere( response < threshold ).tolist() ))
    return coords

#img_max :gaussian image max val
def contrast( dog , img_max, threshold = 0.03 ):
    dog_norm = dog / img_max
    coords = list(map( tuple , np.argwhere( np.abs( dog_norm ) > threshold ).tolist() ))
    return coords






#bta5od lsowr ely byb2o wra b3d

def cube_extrema( img1, img2, img3 ):
    value = img2[1,1]

    if value > 0:
        return all([np.all( value >= img ) for img in [img1,img2,img3]]) # test map
    else:
        return all([np.all( value <= img ) for img in [img1,img2,img3]]) # test map



def dog_keypoints( img_dogs , img_max , threshold = 0.03 ):
    octaves_keypoints = []
    
    for octave_idx in range(N_OCTAVES):
        img_octave_dogs = img_dogs[octave_idx]
        keypoints_per_octave = []
        for dog_idx in range(1, len(img_octave_dogs)-1):
            dog = img_octave_dogs[dog_idx]
            keypoints = np.full( dog.shape, False, dtype = np.bool)
            candidates = set( (i,j) for i in range(1, dog.shape[0] - 1) for j in range(1, dog.shape[1] - 1))
            search_size = len(candidates)
            candidates = candidates & set(corners(dog)) & set(contrast( dog , img_max, threshold ))
            search_size_filtered = len(candidates)
            logger.info('Search size reduced by: {:.1f}%'.format( 100*(1 - search_size_filtered/search_size )))
            for i,j in candidates:
                slice1 = img_octave_dogs[dog_idx -1][i-1:i+2, j-1:j+2]
                slice2 = img_octave_dogs[dog_idx   ][i-1:i+2, j-1:j+2]
                slice3 = img_octave_dogs[dog_idx +1][i-1:i+2, j-1:j+2]
                if cube_extrema( slice1, slice2, slice3 ):
                    keypoints[i,j] = True
            keypoints_per_octave.append(keypoints)
        octaves_keypoints.append(keypoints_per_octave)
    return octaves_keypoints

def dog_keypoints_orientations( img_gaussians , keypoints , num_bins = 36 ):
    kps = []
    for octave_idx in range(N_OCTAVES):
        img_octave_gaussians = img_gaussians[octave_idx]
        octave_keypoints = keypoints[octave_idx]
        for idx,scale_keypoints in enumerate(octave_keypoints):
            scale_idx = idx + 1 ## idx+1 to be replaced by quadratic localization
            gaussian_img = img_octave_gaussians[ scale_idx ] 
            sigma = 1.5 * SIGMA * ( 2 ** octave_idx ) * ( K ** (scale_idx))
            radius = KERNEL_RADIUS(sigma)
            kernel = gaussian_kernel2d(std = sigma, kernlen = 2 * radius + 1)
            gx,gy,magnitude,direction = sift_gradient(gaussian_img)
            direction_idx = np.round( direction * num_bins / 360 ).astype(int)          
            
            for i,j in map( tuple , np.argwhere( scale_keypoints ).tolist() ):
                window = [i-radius, i+radius+1, j-radius, j+radius+1]
                mag_win = padded_slice( magnitude , window )
                dir_idx = padded_slice( direction_idx, window )
                weight = mag_win * kernel 
                hist = np.zeros(num_bins, dtype=np.float32)
                
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
            
                for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():
                    angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
                    kps.append( (i,j,octave_idx,scale_idx,angle))

    return kps


def rotated_subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad
    
    
    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)

def extract_sift_descriptors128( img_gaussians, keypoints, num_bins = 8 ):
    descriptors = []; points = [];  data = {} # 
    for (i,j,oct_idx,scale_idx, orientation) in keypoints:

        if 'index' not in data or data['index'] != (oct_idx,scale_idx):
            data['index'] = (oct_idx,scale_idx)
            gaussian_img = img_gaussians[oct_idx][ scale_idx ] 
            sigma = 1.5 * SIGMA * ( 2 ** oct_idx ) * ( K ** (scale_idx))
            data['kernel'] = gaussian_kernel2d(std = sigma, kernlen = 16)                

            gx,gy,magnitude,direction = sift_gradient(gaussian_img)
            data['magnitude'] = magnitude
            data['direction'] = direction

        window_mag = rotated_subimage(data['magnitude'],(j,i), orientation, 16,16)
        window_mag = window_mag * data['kernel']
        window_dir = rotated_subimage(data['direction'],(j,i), orientation, 16,16)
        window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)

        features = []
        for sub_i in range(4):
            for sub_j in range(4):
                sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                hist = np.zeros(num_bins, dtype=np.float32)
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                features.extend( hist.tolist())
        features = np.array(features) 
        features /= (np.linalg.norm(features))
        np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )
        assert features.shape[0] == 128, "features missing!"
        features /= (np.linalg.norm(features))
        descriptors.append(features)
        points.append( (i ,j , oct_idx, scale_idx, orientation))
    return points , descriptors


def pipeline( input_img ):
    img_max = input_img.max()
    dogs, octaves = image_dog( input_img )
    keypoints = dog_keypoints( dogs , img_max , 0.03 )
    keypoints_ijso = dog_keypoints_orientations( octaves , keypoints , 36 )
    points,descriptors = extract_sift_descriptors128(octaves , keypoints_ijso , 8)
    return points, descriptors





def kp_list_2_opencv_kp_list(kp_list):

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)),
                                 y=kp[0] * (2**(kp[2]-1)),
                                 _size=kp[3],
                                 _angle=kp[4],
#                                  _response=kp[IDX_RESPONSE],
#                                  _octave=np.int32(kp[2]),
                                 # _class_id=np.int32(kp[IDX_CLASSID])
                                 )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list





def draw_matches(img1, kp1, img2, kp2, matches, colors,count): 
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 3
    thickness = 1
    for idx in range(min(count,len(matches))):
        m = matches[idx]
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, colors[idx], thickness)
        cv2.circle(new_img, end1, r, colors[idx], thickness)
        cv2.circle(new_img, end2, r, colors[idx], thickness)
    
    plt.figure(figsize=(20,20))
    plt.imshow(new_img)
    plt.show()
    
    
    
    
    
    
    
def match( img_a, pts_a, desc_a, img_b, pts_b, desc_b):
    img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
    
    desc_a = np.array( desc_a , dtype = np.float32 )
    desc_b = np.array( desc_b , dtype = np.float32 )

    pts_a = kp_list_2_opencv_kp_list(pts_a)
    pts_b = kp_list_2_opencv_kp_list(pts_b)

    # create BFMatcher object
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_a,desc_b,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.25*n.distance:
            good.append(m)

    # Sort them in the order of their distance.
#     matches = sorted(good, key = lambda x:x.distance)
#     distances = np.array(list(map(lambda x:x.distance,matches)),dtype=float)
#     print(distances)
#     distances = (distances - distances.min())/(distances.max()-distances.min())
#     colors = np.array(cm.get_cmap('viridis')(distances)).tolist()
    # cv2.drawMatchesKnn expects list of lists as matches.
#      draw_matches(img_a,pts_a,img_b,pts_b,matches,colors,30)
    img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
#     cv2.drawMatchesKnn(img_a,pts_a,img_b,pts_b,good,outImg = img_match, matchColor=None,
#                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.figure(figsize=(20,20))
#     plt.imshow(img_match)
#     plt.show()
    cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20,20))
    plt.imshow(img_match)
    plt.axis('off')
    plt.savefig("SIVT1.jpg")
    plt.show() 
    
    
    
    
    
    
img = np.array(Image.open('img.jpg'))
img,r=cvutils_ass3.sift_resize(img)
img_gry=cvutils_ass3.rgb2gray(img)

points1,desc1=pipeline(img_gry)

img2 = np.array(Image.open('02.jpg'))
img2,_=cvutils_ass3.sift_resize(img2,r)
img_gry2=cvutils_ass3.rgb2gray(img2)
points2,desc2=pipeline(img_gry2)
print("6")

match(img,points1,desc1,img2,points2,desc2)
