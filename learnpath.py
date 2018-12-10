import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import cv2
import math
from sklearn import svm
import sklearn.metrics as metric
from sklearn.cluster import KMeans
from fractions import Fraction

data = pd.read_csv("Path.csv")
track_times = data.groupby('n').t.count()

#plot each observation length v time

def plotLT():#plot length of curve v time
    times = []
    lengths = []
    minTime = 2000 #adjust to remove noise
    for i in track_times.loc[track_times>minTime].index:
        times.append(track_times[i])
        lengths.append(computeLength(i))
        #displacements.append(computeDisplacement(i))
        i += 1
        
    plt.scatter(times,lengths)
    plt.xlabel('time')
    plt.ylabel('length')
    plt.show()
    return times,lengths;

times = []
disp = []
id = []
def plotDT():#plot distance v time

    minTime = 69 #adjust to remove noise
    for i in track_times.loc[track_times>minTime].index:
        times.append(track_times[i])
        disp.append(computeDisplacement(i))
        id.append(i)
        #displacements.append(computeDisplacement(i))
        i += 1
        
    plt.scatter(times,disp)
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.show()

def toPandas(id, x_arr, y_arr):
    new = [('n',id),
           ('time',x_arr),
           ('displacement',y_arr)
          ]
    workingdf = pd.DataFrame.from_items(new)
    return workingdf;
    
#compute length
def computeLength(i):
    coor = data.loc[data.n == i] # access all x,y, points
    #length = summation of sqrt((x i-x i-1)^2 + (y i- y i-1)^2)
    total_l = 0
    q = coor.index[1]
    while q <= coor.index[coor.x.count()-1]: 
        x2 = float(coor.x[q])
        x1 = float(coor.x[q-1])
        y2 = float(coor.y[q])
        y1 = float(coor.y[q-1])
        
        l = math.hypot(x2 - x1, y2 - y1)
        total_l += l
        q += 1

    return total_l;

def showPath(i):#Points gathered are not perfect, fine a way to smooth out
    x = data.loc[data.n == i, 'x']
    y = data.loc[data.n == i, 'y']
    
    plt.scatter(x,y)
    plt.gca().invert_yaxis()
    plt.show()

def computeDisplacement(i):
    coor = data.loc[data.n == i, 'x':]
    fi = coor.index[0]
    li = coor.index[coor.x.count()-1]
    x1 = coor.x[fi]
    x2 = coor.x[li]
    y1 = coor.y[fi]
    y2 = coor.y[li]
    
    disp = math.hypot(x2 - x1, y2 - y1)
    
    return disp;

#Compute curvature
def computeCurvature(i):
    coor = data.loc[data.n == i, 'x':]
    #curvature = xi*yii-yi*xii/(xi^2+yi^2)^2/3
    curve = []
    q = coor.index[2]
    while q <= coor.index[coor.x.count()-1]: 
        xii = coor.x[q]-2*coor.x[q-1]+coor.x[q-2]
        xi = coor.x[q]-coor.x[q-1]
        yii = coor.y[q]-2*coor.y[q-1]+coor.y[q-2]
        yi = coor.y[q]-coor.y[q-1]
        
                
        num = xi*yii-yi*xii
        den = float((xi**2 + yi**2)**(1.5))
        

        cT = num/den
        
        if cT == nan
            curve.append(0)
        else
            curve.append(cT)
        print(q,num,den,cT)
        q += 1
     
    #histogram
    #n, bins, patches = plt.hist(coor, 3, facecolor='blue', alpha=0.5)
    #plt.show()
    print(i)
    plt.plot(curve)
    plt.show()
    
    

#recognize kart by knn by t, and length of function? SVM?
def classifySVM(x_arr, y_arr):#SVM requires LABELLED input data, 
    svc = svm.SVC(kernel='linear').fit(x_arr,y_arr)
    svc.get_params(true)
 
def classifyKMeans(dataframe):
    #actual K-Means classification
    X = dataframe.loc[:, 'time':]#convert raw dataframe into usable param
    kmeans = KMeans(n_clusters = 2).fit(X) #kmeans classification
    result = kmeans.labels_ #labels per data point
    df_classified = dataframe.assign(label = result)#labels appended to working dataframe
    
    return df_classified;
        
def houghline(): #this works, identifies the edges of the track
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
#Markov Model?








































