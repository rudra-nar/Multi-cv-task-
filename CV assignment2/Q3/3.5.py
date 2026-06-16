import cv2
import numpy as np
import os


internal = np.array([[956.63717905, 0., 369.05488009], 
                     [0., 957.55139544, 651.41419152],
                     [0., 0., 1.]])
distortion = np.array([[0.19763675, -0.70686389, 0.00336937, 0.00698244, 0.04877254]])
CHECKERBOARD = (8,6)


#coustom
internal = np.array([[1.30516427e+03, 0.00000000e+00, 7.67886336e+02], [0.00000000e+00, 1.30585706e+03, 5.64200520e+02], [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 0.17404508 ,-0.65946933, -0.00424748 ,-0.00779977 , 0.70234783]])
CHECKERBOARD = (9,6)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

data_path = r"C:\Users\Asus\Documents\CV assignment2\chessboard_dataset"
data_path = r"C:\Users\Asus\Documents\CV assignment2\custom_chessboard"
image_files = os.listdir(data_path)[:25]

for i, fname in enumerate(image_files):
    img = cv2.imread(os.path.join(data_path, fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
       
        _, rvec, tvec = cv2.solvePnP(objp, corners, internal, distortion)
        
   
        reprojected_pts, _ = cv2.projectPoints(objp, rvec, tvec, internal, distortion)

        for pt in corners:
            cv2.circle(img, tuple(pt[0].astype(int)), 5, (0, 0, 255), -1) 

        for pt in reprojected_pts:
            cv2.circle(img, tuple(pt[0].astype(int)), 3, (0, 255, 0), -1)

        cv2.putText(img, f"Img {i+1}: Red=Detected, Green=Projected", (30,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow('Reprojection Comparison', img)
        cv2.waitKey(0)
cv2.destroyAllWindows()