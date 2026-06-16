## point = intrensic * extrnal * world point
import cv2
import numpy as np
import os


internal = np.array([[956.63717905,0.,369.05488009], [ 0.,957.55139544, 651.41419152],[0.,0.,1.]])
dist = np.array([[ 0.19763675 ,-0.70686389,  0.00336937,  0.00698244,  0.04877254]])

#custom dataset
#internal = np.array([[1.30516427e+03, 0.00000000e+00, 7.67886336e+02], [0.00000000e+00, 1.30585706e+03, 5.64200520e+02], [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[ 0.17404508 ,-0.65946933, -0.00424748 ,-0.00779977 , 0.70234783]])

CHECKERBOARD = (8,6)
#CHECKERBOARD = (9,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

data_path = r"C:\Users\Asus\Documents\CV assignment2\chessboard_dataset"
#data_path = r"C:\Users\Asus\Documents\CV assignment2\custom_chessboard"

for images in os.listdir(data_path)[0:2]:
    img = cv2.imread(os.path.join(data_path,images))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        success, rvec, tvec = cv2.solvePnP(objp, corners2, internal, dist)
        matrix, jacobian = cv2.Rodrigues(rvec)
        normal = matrix[:,2]
        
        print(matrix)
        print('===============')
        print(tvec)
        print('===============')


