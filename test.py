# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:30:05 2018

@author: Emman-1207
"""

#extract track?
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
def edgeDetect():
    img = cv2.imread('RCtrack.png')

    gsimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(gsimg,(5,5),2)
    
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)  
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7) 

    plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.imshow(laplacian,cmap = 'gray')
    plt.title('laplacian'), plt.xticks([]), plt.yticks([])

    plt.show()
    cv2.imwrite('sobelytest.png', sobely)
    cv2.imwrite('sobelxtest.png',sobelx)
    cv2.imwrite('laptest.png',laplacian)
    return;
    
def canny():
    img = cv2.imread('RCtrack.png',0)
    edges = cv2.Canny(img,0,142)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    cv2.imwrite('edges.png',edges)

def houghline(): #this works
    img = cv2.imread("RCtrack.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, 120)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, maxLineGap=70)
 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
 
    cv2.imshow("Edges", edges)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()