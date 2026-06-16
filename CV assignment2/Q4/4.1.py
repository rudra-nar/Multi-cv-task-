
##h ttps://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
import cv2 as cv
import os

path = r"C:\Users\Asus\Documents\CV assignment2\panorama_dataset"

img1 = cv.imread(os.path.join(path,"image1.png"))
img2 = cv.imread(os.path.join(path,"image2.png"))

gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)


sift = cv.SIFT_create()
kp1 = sift.detect(gray1,None)
kp2 = sift.detect(gray2,None)

img1=cv.drawKeypoints(gray1,kp1,img1)
img2=cv.drawKeypoints(gray2,kp2,img2)

cv.imshow("Keypoints1", img1)
cv.imshow("Keypoints2", img2)

cv.waitKey(0)