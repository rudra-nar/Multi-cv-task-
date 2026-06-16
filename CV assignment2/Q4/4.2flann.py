import cv2 as cv
import os
#https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

path = r"C:\Users\Asus\Documents\CV assignment2\panorama_dataset"
img1 = cv.imread(os.path.join(path,"image1.png"))
img2 = cv.imread(os.path.join(path,"image2.png"))


  
sift = cv.SIFT_create()
key_point1, descr1 = sift.detectAndCompute(img1,None)
key_point2, descr2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=1000) 

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descr1,descr2,k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask,flags = 2)

    # drawing nearest neighbours
img = cv.drawMatchesKnn(img1,key_point1,img2,key_point2,matches,None, **draw_params)

cv.imshow("Keypoints2", img)

cv.waitKey(0)