import cv2 as cv
import numpy as np

image = cv.imread('car.jpg')
image1 = cv.imread('car1.jpg')

sift = cv.xfeatures2d.SIFT_create()
gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
kp1, desc1 = sift.detectAndCompute(gray1, None)
print(np.array(desc1).flatten())

gray2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kp2, desc2 = sift.detectAndCompute(gray2, None)
print(np.array(desc2).flatten())

image_flatten = sorted(cv.resize(image, (32, 32)).flatten())
image1_flatten = sorted(cv.resize(image1, (32, 32)).flatten())
print(image_flatten)
print(image1_flatten)

import matplotlib.pyplot as plt

plt.plot(np.array(desc2).flatten())
plt.show()

plt.plot(np.array(desc1).flatten())
plt.show()

print("Hello1")