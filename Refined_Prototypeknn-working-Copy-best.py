import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from matplotlib import style
import datetime
import os
from trackerpy import CentroidTracker
from scipy import optimize

knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,12))
es1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
camera = cv2.VideoCapture("1train1.mp4") #place video filename here
font = cv2.FONT_HERSHEY_SIMPLEX
xpoints = []
ypoints = []
rects = []
objectLabel = []
ntxy = []
ct = CentroidTracker()
(H, W) = (None, None)
frame_num = -2

style.use('fivethirtyeight')
def col(i):
    C = [(0,0,255),(0,255,255),(255,127,0),(123,255,0),(255,0,255),(27,152,214),(140,0,255),(226,97,179),(237,255,0)]
    return C[i%len(C)]

#Identify region of interest, draw centroid, and add store points
def drawCnt(fn, cnt):
    if cv2.contourArea(cnt) > 150:
        (x, y, w, h) = cv2.boundingRect(cnt) #draw ROI
        cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)
        (a,b) = ((x+(w/2)),(y+(h/2))) #draw centroid
        cv2.circle(frame, (a,b) , 5, (255, 0, 0), cv2.FILLED)
        rects.append([x,y,x+w,y+h])#add to current bounding boxes

def color(frame,rects):
    color = []
    for ROI in rects:
        color.append(frame[ROI[1]:ROI[3], ROI[0]:ROI[2]])
    return color

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#creating ntxy file and updating the file
def recordingNTXY(R):
    try:
        os.rename('Path.ntxy',datetime.datetime.now().strftime("%Y%m%d-%H%M")+'H.csv')
    except:
        print 'error creating backup'
    with open('Path.csv','w') as file:
        file.write('n,t,x,y' + '\n')
        for row in R:
            if len(row)<5:
                file.write(','.join(map(str, row))+"\n")

#Main Method
while True:
  frame_num = frame_num + 1
  ret, frame = camera.read()
  if not ret:
    break
  #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)  
  if W is None or H is None:#get shape of frame 
		(H, W) = frame.shape[:2]
  
  gamma = 2.0
  for gamma in np.arange(0.0, 3.5, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
   if gamma == 1:
		continue
 
   # apply gamma correction and show the images
   gamma = gamma if gamma > 0 else 0.1
   adjusted = adjust_gamma(frame, gamma=gamma)
   cv2.imshow("Images", np.hstack([frame, adjusted]))
  fg = knn.apply(adjusted.copy()) #this is where the actual image processing happens, erosion before dilation may be useful here(opening)	
  fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
  bw_and = cv2.bitwise_and(fg_bgr, frame)
  draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
  draw = cv2.GaussianBlur(draw, (21, 21), 0)
  draw = cv2.threshold(draw, 20, 255, cv2.THRESH_BINARY)[1]
  draw = cv2.dilate(draw, es1, iterations = 2)
  draw = cv2.erode(draw, es1, iterations = 4)
  draw = cv2.dilate(draw, es1, iterations = 3)
  cv2.imshow("dilate", draw)
  image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

  for c in contours:
      drawCnt(frame, c)

#get the color of each object
  colors_of_objects = color(frame,rects)
# update our centroid tracker using the computed set of bounding box rectangles
  objects = ct.update(rects, colors_of_objects)
# loop over the tracked objects
  for (objectID, centroid) in objects.items():
    # draw both the ID of the object on the output frame
    text = "ID {}".format(objectID) + " {}".format(centroid[0]) + " {}".format(centroid[1])
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
    xpoints.append(centroid[0])#x for scatter plot
    ypoints.append(centroid[1])#y for scatter plot
    objectLabel.append(objectID)#ID for scatter plot
    ntxy.append([objectID, frame_num,centroid[0],centroid[1]])#format: carID, frame, x, y
    cv2.imshow("video", frame)
   
    #for sorting ntxy based on the car ID, SOURCE https://stackoverflow.com/questions/17555218/python-how-to-sort-a-list-of-lists-by-the-fourth-element-in-each-list
    ntxy.sort(key=lambda x: int(x[0]))
  del rects[:]#delete bounding boxes of current frame
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
      break

recordingNTXY(ntxy)
print("saving ntxy file...")

cv2.destroyAllWindows()