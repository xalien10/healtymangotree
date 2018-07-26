import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('affected/a97c2088-9bf7-4840-8a41-813aadae4aa6_JR_FrgE.S_2793.JPG')
image2 = cv2.imread('affected/0bc40cc3-6a85-480e-a22f-967a866a56a1_JR_FrgE.S_2784.JPG')


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


image1_gray = to_gray(image1)
image2_gray = to_gray(image2)

plt.imshow(image1, cmap='gray')

image1_kp, image1_desc = gen_sift_features(image1)
image2_kp, image2_desc = gen_sift_features(image2)

print('Here are what our SIFT features look like for the front-view octopus image:')
show_sift_features(image1_gray, image1, image1_kp)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(image1_desc, image2_desc)

# Sort the matches in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# draw the top N matches
N_MATCHES = 100

match_img = cv2.drawMatches(
    image1, image1_kp,
    image2, image2_kp,
    matches[:N_MATCHES], image2.copy(), flags=0)

plt.figure(figsize=(12, 6))
plt.imshow(match_img)
