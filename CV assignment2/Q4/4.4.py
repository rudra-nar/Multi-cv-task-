import cv2 as cv
import os
import numpy as np
#https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

path = r"C:\Users\Asus\Documents\CV assignment2\panorama_dataset"
img1 = cv.imread(os.path.join(path,"image1.png"))
img2 = cv.imread(os.path.join(path,"image2.png"))


M1 = np.loadtxt(r"C:\Users\Asus\Documents\CV assignment2\homography1to2.csv", delimiter=',')
M2 = np.loadtxt(r"C:\Users\Asus\Documents\CV assignment2\homography2to1.csv", delimiter=',')

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]
size = (w1 + w2, h1+ h2)

offset = np.array([[1, 0,100], [0, 1, 100], [0, 0, 1]])
offset2 = np.array([[1, 0, 300], [0, 1, 100], [0, 0, 1]])

img2_img1 = cv.warpPerspective(img2, offset @  M2, size)
cv.imshow("img21", img2_img1)
#cv.imshow("img1", img1)


img1_img2 = cv.warpPerspective(img1, offset2 @ M1, size)
cv.imshow("Warped Image 1", img1_img2)
#cv.imshow("img2", img2)

cv.waitKey(0)
cv.destroyAllWindows()


# src: input image
# M: Transformation matrix
# dsize: size of the output image
# flags: interpolation method to be used