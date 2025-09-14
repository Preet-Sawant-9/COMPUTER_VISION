import cv2
import numpy as np

img = cv2.imread('DUBAI.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
Ixx = Ix ** 2
Iyy = Iy ** 2
Ixy = Ix * Iy
Ixx = cv2.GaussianBlur(Ixx, (5, 5), 0)
Iyy = cv2.GaussianBlur(Iyy, (5, 5), 0)
Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)
k = 0.04
R = (Ixx * Iyy - Ixy ** 2) - k * (Ixx + Iyy) ** 2
threshold = 0.01 * R.max()
img[R > threshold] = [0, 0, 255]
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
