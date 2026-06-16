import cv2 as cv
import os
import numpy as np
#https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

path = r"C:\Users\Asus\Documents\CV assignment2\panorama_dataset"
img1 = cv.imread(os.path.join(path,"image1.png"))
img2 = cv.imread(os.path.join(path,"image2.png"))


  
sift = cv.SIFT_create()
kp1, descr1 = sift.detectAndCompute(img1,None)
kp2, descr2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=1000) 

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descr1,descr2,k=2)

matchesMask = [[0, 0] for i in range(len(matches))]


good=[]
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M1, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
M2, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC,5.0)

np.savetxt("homography1to2.csv",M1,delimiter=',')
np.savetxt("homography2to1.csv",M2,delimiter=',')

