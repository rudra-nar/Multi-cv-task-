##h ttps://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
import cv2 as cv
import os

path = r"C:\Users\Asus\Documents\CV assignment2\panorama_dataset"

img1 = cv.imread(os.path.join(path,"image1.png"))
img2 = cv.imread(os.path.join(path,"image2.png"))

gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)


sift = cv.SIFT_create()
kp1,decription1 = sift.detectAndCompute(gray1,None)
#print(decription1[1])
kp2,decription2 = sift.detectAndCompute(gray2,None)

matching={}
print(len(decription1))
mached = [] 
for j in range(len(decription1)):
    p = decription1[j]
    min = 100000000000000000000000000000
    min_idx=0
    #print(j)
    for k in range(len(decription2)):
        if k in mached:
            continue
        q = decription2[k]
        dis = 0
        for i in range(len(p)):
            dis += (p[i]-q[i])**2
        if dis < min:
            min = dis
            min_idx = k

    matching[kp1[j]] = kp2[min_idx]
    mached.append(min_idx)

    
#print(matching)
    
