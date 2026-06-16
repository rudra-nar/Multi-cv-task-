import cv2 as cv
import os
import numpy as np

path = r"C:\Users\Asus\Documents\CV assignment2"
data_path = os.path.join(path, "panorama_dataset")

img1 = cv.imread(os.path.join(data_path, "image1.png"))
img2 = cv.imread(os.path.join(data_path, "image2.png"))

M1 = np.loadtxt(os.path.join(path, "homography1to2.csv"), delimiter=',')
M2 = np.loadtxt(os.path.join(path, "homography2to1.csv"), delimiter=',')

offset = np.array([[1, 0, 300], [0, 1, 300], [0, 0, 1]], dtype=np.float32)
size = (img1.shape[1] + img2.shape[1] + 600, img1.shape[0] + 600)

img1to2 = cv.warpPerspective(img1, offset @ M1, size)
img2_warped = cv.warpPerspective(img2, offset, size)

img2to1 = cv.warpPerspective(img2, offset @ M2, size)
img1_warped = cv.warpPerspective(img1, offset, size)

panorama_raw1 = np.where(img2_warped == 0, img1to2, img2_warped)
panorama_raw2 = np.where(img1_warped == 0, img2to1, img1_warped)

cv.imshow("Raw Panorama1", panorama_raw1)
cv.imshow("Raw Panorama2", panorama_raw2)

cv.waitKey(0)
cv.destroyAllWindows()