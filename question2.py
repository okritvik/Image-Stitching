"""
Created on Tue Mar 15 21:51:36 2022

@author: okritvik
"""

import cv2 #USING OPENCV 3.4
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("./images/Q2imageA.png")
image2 = cv2.imread("./images/Q2imageB.png")

#Extracting Sift Features

sift = cv2.xfeatures2d.SIFT_create()

# Getting the key points and descriptors
kp1, disc1 = sift.detectAndCompute(image1,None)
kp2, disc2 = sift.detectAndCompute(image2,None)

# Creating the bf matcher object
bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck = True)

# Getting the match points using the descriptors
matches = bf.match(disc1,disc2)

# Sorting them according to the distance
matches = sorted(matches, key=lambda x:x.distance)
src = []
dst = []
count = 0

#Getting some random points which gave good homography
for match in matches:
    if(count>30):    
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        src.append(p1)
        dst.append(p2)
        count += 1
        if(count>35):
            break
    else:
        count += 1

# print(src)
# print(dst)

#Finding the homography
Homography = cv2.findHomography(np.array(dst), np.array(src))
H = Homography[0]

size1 = image1.shape
size2 = image2.shape

# Drawing the matches using the features
image3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[30:34], image2)
plt.imshow(cv2.cvtColor(image3,cv2.COLOR_BGR2RGB))
plt.title("Matched Features using SIFT")
plt.show()

# Warping the second image using the homography matrix
final = cv2.warpPerspective(image2, H, ((image1.shape[1] + image2.shape[1]), image2.shape[0])) #wraped image
plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.show()

# Stitching the images together
try:
    for a in range(size1[0]):
        for b in range(size1[1]):
            final[a][b] = image1[a][b]
            # f = [a,b,1]
            # f = np.reshape(f,(3,1))
            # x, y, z = np.matmul(H,f)
            # warped[int(y/z)][int(x/z)] = image2[a][b]
            
except:
    pass
# Show the stitched image
plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.show()
