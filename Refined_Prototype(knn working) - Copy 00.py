import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import os
from trackerpy import CentroidTracker


knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,12))
es1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
camera = cv2.VideoCapture("1train1.MP4") #place video filename here
font = cv2.FONT_HERSHEY_SIMPLEX
xpoints = []
ypoints = []
i=0
rects = []
objectLabel = []
ntxy = []
ct = CentroidTracker()
(H, W) = (None, None)

style.use('fivethirtyeight')
def col(i):
    C = [(0,0,255),(0,255,255),(255,127,0),(123,255,0),(255,0,255),(27,152,214),(140,0,255),(226,97,179),(237,255,0)]
    return C[i%len(C)]

#Identify region of interest, draw centroid, and add store points
def drawCnt(fn, cnt):
    if cv2.contourArea(cnt) > 1400:
        (x, y, w, h) = cv2.boundingRect(cnt) #draw ROI
        cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)
        (a,b) = ((x+(w/2)),(y+(h/2))) #draw centroid
        cv2.circle(frame, (a,b) , 5, (255, 0, 0), cv2.FILLED)
        rects.append([x,y,x+w,y+h])#add to current bounding boxes

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
  ret, frame = camera.read()
  if not ret:
    break
  #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)  
  if W is None or H is None:#get shape of frame 
		(H, W) = frame.shape[:2]
  fg = knn.apply(frame.copy()) #this is where the actual image processing happens, erosion before dilation may be useful here(opening)
  fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
  bw_and = cv2.bitwise_and(fg_bgr, frame)
  draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
  draw = cv2.GaussianBlur(draw, (21, 21), 0)
  draw = cv2.threshold(draw, 30, 255, cv2.THRESH_BINARY)[1]
  draw = cv2.dilate(draw, es1, iterations = 2)
  draw = cv2.erode(draw, es1, iterations = 4)
  draw = cv2.dilate(draw, es1, iterations = 3)
  cv2.imshow("dilate", draw)
  image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

  for c in contours:
      drawCnt(frame, c)

# update our centroid tracker using the computed set of bounding box rectangles
  objects = ct.update(rects)
# loop over the tracked objects
  for (objectID, centroid) in objects.items():
    # draw both the ID of the object on the output frame
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    xpoints.append(centroid[0])#x for scatter plot
    ypoints.append(centroid[1])#y for scatter plot
    objectLabel.append(objectID)#ID for scatter plot
    ntxy.append([objectID, i,centroid[0],centroid[1]])#format: carID, frame, x, y
    #for sorting ntxy based on the car ID, SOURCE https://stackoverflow.com/questions/17555218/python-how-to-sort-a-list-of-lists-by-the-fourth-element-in-each-list
    ntxy.sort(key=lambda x: int(x[0]))
  #frame = cv2.resize(frame, None, fx=0.4, fy=0.4) 
  cv2.imshow("Contour", frame)
  i=i+1
  del rects[:]#delete bounding boxes of current frame
  if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
      break

print("saving ntxy file...")
recordingNTXY(ntxy)

#SOURCE for plotting https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
N = len(np.unique(objectLabel))
# setup the plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
# define the data
x = xpoints
y = ypoints
tag = objectLabel # Tag each point with a corresponding label    

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# make the scatter
scat = ax.scatter(x,y,c=tag, cmap=cmap, norm=norm)
# create the colorbar
#cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
img = plt.imread("RCtrack.png") #place screencap of track here
plt.imshow(img,zorder=0)
plt.show()
cv2.destroyAllWindows()