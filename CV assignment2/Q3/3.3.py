## 
import cv2
import numpy as np
import os

internal = np.array([[956.63717905,0.,369.05488009], [ 0.,957.55139544, 651.41419152],[0.,0.,1.]])
distortion = np.array([[ 0.19763675 ,-0.70686389,  0.00336937,  0.00698244,  0.04877254]])

#coustom
internal = np.array([[1.30516427e+03, 0.00000000e+00, 7.67886336e+02], [0.00000000e+00, 1.30585706e+03, 5.64200520e+02], [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
distortion = np.array([[ 0.17404508 ,-0.65946933, -0.00424748 ,-0.00779977 , 0.70234783]])



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = [] 

prev_img_shape = None
data_path = r"C:\Users\Asus\Documents\CV assignment2\chessboard_dataset"
data_path = r"C:\Users\Asus\Documents\CV assignment2\custom_chessboard"

coeff = []
for images in os.listdir(data_path)[0:5]:
    img = cv2.imread(os.path.join(data_path,images))
    dst = cv2.undistort(img,internal ,distortion, None)
    coeff.append[dst]
    cv2.imshow('undistorted',np.hstack((dst,img)))
    cv2.waitKey(0)

        
