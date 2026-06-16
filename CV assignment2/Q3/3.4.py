import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

CHECKERBOARD = (8,6)

#CHECKERBOARD = (9,6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


internal = np.array([[956.63717905,0.,369.05488009], [ 0.,957.55139544, 651.41419152],[0.,0.,1.]])
dist = np.array([[ 0.19763675 ,-0.70686389,  0.00336937,  0.00698244,  0.04877254]])

#custom
#internal = np.array([[1.30516427e+03, 0.00000000e+00, 7.67886336e+02], [0.00000000e+00, 1.30585706e+03, 5.64200520e+02], [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[ 0.17404508 ,-0.65946933, -0.00424748 ,-0.00779977 , 0.70234783]])



objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

data_path = r"C:\Users\Asus\Documents\CV assignment2\chessboard_dataset"

#data_path = r"C:\Users\Asus\Documents\CV assignment2\custom_chessboard"

errors = []
imgages = []
i = 1

for filename in os.listdir(data_path)[:25]:
    img = cv2.imread(os.path.join(data_path, filename))
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        success, rvec, tvec = cv2.solvePnP(objp, corners2, internal, dist)
        
        if success:
            imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, internal, dist)
            
            error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
            imgages.append("img"+str(i))

            for pt in corners2:
                cv2.circle(img, tuple(pt[0].astype(int)), 5, (0, 0, 255), -1)

            for pt in imgpoints2:
                cv2.circle(img, tuple(pt[0].astype(int)), 2, (0, 255, 0), -1)
            
            cv2.imshow('Red=Detected, Green=Projected', img)
            cv2.waitKey(0)
    i += 1

cv2.destroyAllWindows()

# Statistics
print(f"Mean Error: {np.mean(errors)}")
print(f"Std Dev: {np.std(errors)}")

plt.figure(figsize=(10,5))
plt.bar(imgages, np.array(errors), color='skyblue')
plt.title("Re-projection Error per Image")
plt.show()