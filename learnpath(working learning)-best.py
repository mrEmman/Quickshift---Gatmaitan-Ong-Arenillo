import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import svm
from sklearn.cluster import KMeans
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from numpy import cos, sin, pi, absolute, arange, fft
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Path.csv")
track_times = data.groupby('n').t.count()

#plot each observation length v time

def plotLT():#plot length of curve v time
    times = []
    lengths = []
    minTime = 25 #adjust to remove noise
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


#these are inputs into method toPandas()
times = []
disp = []
id = []
def plotDT():#plot distance v time

    minTime = 250 #adjust to remove noise
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

def toPandas(id, x_arr, y_arr): #takes arrays from raw dataframe into more useful and more concise working dataframe
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
        x2 = coor.x[q]
        x1 = coor.x[q-1]
        y2 = coor.y[q]
        y1 = coor.y[q-1]
        
        l = math.hypot(x2 - x1, y2 - y1)
        total_l += l
        q += 1
    
    return total_l;

def showPath(i):#shows the actual path taken, not used in computation is more of a manual double check but may be useful for getting path from the model
    x = data.loc[data.n == i, 'x']
    y = data.loc[data.n == i, 'y']

    print(i)
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
        den = float(xi**2 + yi**2)**1.5
        
        if den == 0:
            cT = 0
        else:
            cT = num/den
        
        curve.append(cT)
        q += 1
        
    return curve
    

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
        
def lpFilter(curve): #takes array/list from computeCurvature() removes noise 
    fc = 0.1
    b = 0.08
    N = int(np.ceil(4/b))
    if not N % 2: N += 1
    n = np.arange(N)
    
    h = np.sinc(2 * fc * (n- (N - 1) / 2.))
    
    w = 0.42 - 0.5 * np.cos(np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    h = h * w
    
    h = h / np.sum(h)
    
    s_filtered = np.convolve(curve,h)
    
    return s_filtered #returns filtered curvature(list)

def fourierExtrapolation(x, n_predict): #takes signal, extrapolates n points from end of signal
    n = x.size
    n_harm = 18                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = range(n)
    # sort frequency indexes by amplitude size, higher -> lower
    indexes.sort(key = lambda i: np.absolute(x_freqdom[i]) / n, reverse = True)

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
    
def generateModel(signal): #takes filtered curvatures from lpFilter() as arg
    x = signal
    n_predict = len(x)
    extrapolation = fourierExtrapolation(x, n_predict)
    #plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
    #plt.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
    #plt.legend()
    #plt.savefig('forcast.png')
    
    model = extrapolation[n_predict:-1]
    
    return model

def check(predicted,expected): #takes filtered curvatures from lpFilter()
    plt.plot(expected,'-r')
    plt.plot(predicted,'-b')
    
    error = []
    i = 0
    
    if len(predicted) > len(expected):#prevents out of bounds
        stop = len(expected)
    else:
        stop = len(predicted)
    
    while i < stop-1:
        e = expected[i]-predicted[i]
        error.append(e)
        i+=1
        
    #plt.savefig('boi.png')
    return error #returns the error between predicted and expected

def correct(model,error, test):#filtered test curvature
    mod = np.array(model)
    err = np.array(error)
    
    threshold = 0.01 #error threshold 
    
    correction = np.where(abs(err) > threshold) #indexes beyond accuracy threshold

    new = generateModel(test) #re-train

    #get corrected portions from re-trained model, place into existing model
    for c in correction:
        mod[c] = new[c]
    
    return mod
    
#====================================Main Method========================================
groundtruth = pd.read_csv('GroundTruthData.csv')
#manually identified good paths
good = [195,192,186,175,174,164,156,155,144,135,134]

def main(df):#takes list of n from the ntxy
    ns = df
    mod = generateModel(lpFilter(computeCurvature(ns[0]))) #generates initial model
    i = 0
    while i <= len(ns)-1: #iterates through all good paths
        if i == len(ns)-1: #if in last index compare to first
            test = lpFilter(computeCurvature(ns[0])) 
        else: #compare to next index
            test = lpFilter(computeCurvature(ns[i]))    
        err = check(mod,test) #compute for error
        improved = correct(mod,err,test) # correct model
        i+=1
    
    return improved
    

def clean(df): #removes paths which collide with the walls and are below average time
    ct = df.loc[df.time < 300]
    cd = ct.loc[ct.displacement > 1000]

    return cd
  
plotDT()

#below is just data type conversion so its easier for the methods i made to handle
df = toPandas(id,times,disp)
df = clean(df)
dfn = df.n
ns = []
for n in dfn:
    ns.append(int(n))
test = ns[16:]
    
    
    
    
    
    
    
    
    





































