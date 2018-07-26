import cv2
import sys
import os.path
import numpy as np


def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0, 1), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0, 1), 1)

    return out


def compare(filename1, filename2):
    img1 = cv2.imread(filename1)  # queryImage
    img2 = cv2.imread(filename2)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda val: val.distance)

    img3 = drawMatches(img1, kp1, img2, kp2, matches[:25])

    # Show the image
    cv2.imshow('Matched Features', img3)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')


# if len(sys.argv) != 3:
#     sys.stderr.write("usage: compare.py <queryImageFile> <sourceImageFile>\n")
#     sys.exit(-1)

# compare('input/01.jpg', 'sift_keypoints.jpg')
# compare('input/01.jpg', '01.jpg')
compare('affected/01a66316-0e98-4d3b-a56f-d78752cd043f_FREC_Scab_3003.JPG',
        'affected/0c620ec5-11cf-4120-94ab-1311e99df147_FREC_Scab_3131.JPG')
# image1 = cv2.imread('affected/a97c2088-9bf7-4840-8a41-813aadae4aa6_JR_FrgE.S_2793.JPG')
# image2 = cv2.imread('affected/0bc40cc3-6a85-480e-a22f-967a866a56a1_JR_FrgE.S_2784.JPG')
# compare('output/image_01.png', 'output/image_11.png')
