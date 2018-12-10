import cv2, numpy as np

def col(i):
    C = [(0,0,255),(0,255,255),(255,127,0),(123,255,0),(255,0,255),(27,152,214),(140,0,255),(226,97,179),(237,255,0)]
    return C[i%len(C)]

import Tkinter, tkMessageBox,tkSimpleDialog, os, datetime, json

def decom(fname):
    fn = fname.split('.')[0]
    i=0
    cap = cv2.VideoCapture(fn+'.mp4')
    while cap.isOpened():
        ret, frame = cap.read()        
        if ret==True:
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
            cv2.imwrite(fn+"{0:0>5}".format(i)+'.jpg',frame)
            if i%500 is 0:
                cv2.imshow('win',frame)
            i=i+1
            if (cv2.waitKey(1) & 0xFF) in [27,ord('q')]:
                break
    print i,'frames extracted'
    cap.release()
    cv2.destroyAllWindows()

def mapModel(fn):
    fn = fn.split(' ')[0]

    Img = {}
    Img[0] = cv2.imread(fn+"{0:0>5}".format(0)+'.jpg')        
    Img[1] = cv2.imread(fn+'.jpg')  
    sh,_,_ = Img[0].shape
    mh,_,_ = Img[1].shape    
    ratio = .9*sh/mh
    Img[1] = cv2.resize(Img[1],None,fx=ratio, fy=ratio)
    
    Lst = [[],[]]

    Win = ['original perspective','target perspective',0]
    for i in range(2):
        cv2.imshow(Win[i],Img[i])

    def end(k):
        root = Tkinter.Tk()
        root.withdraw()
        if idx() == 0:
            answer = tkMessageBox.askyesnocancel("Terminating", "Compute and save homography matrix H?")
            if answer is True:
                H,_ = cv2.findHomography(np.array(Lst[0]),np.array(Lst[1]))
                try:
                    os.rename(fn+'.map',datetime.datetime.now().strftime("%Y%m%d-%H%M")+'H.map')
                except:
                    print 'error creating backup'
                with open(fn+'.map','w') as file:
                    json.dump(H.tolist(),file)
                print 'Matrix saved'
            elif answer is None:
                k = 0
        else:
            answer = tkMessageBox.askokcancel("Warning", "Mapping not complete. Are you sure you want to exit?")
            if answer is not True:
                k = 0
        root.destroy()
        return k

    def undo():
        if Win[2] > 0:
            del Lst[idx(-1)][-1]
            cv2.setMouseCallback(Win[idx()], onMseClk)            
            print 'last item removed'
        else:
            print 'Cant undo anymore'

    def idx(i=0):
        Win[2] += i
        return Win[2]%2
    
    def plotAndShow(event,xx=None,yy=None):
        im = Img[idx()].copy()
        for k,v in enumerate(Lst[idx()]):
            x,y = v
            cv2.putText(im,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.5,col(k))            
            cv2.circle(im, (x,y),2,col(k), -1)  
        if event == 1:
            _,w,_ = im.shape
            cv2.putText(im,str(xx)+','+str(yy),(w-200,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(10,200,0))
        cv2.imshow(Win[idx()],im)
                                
    def onMseClk(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            Lst[idx()].append((x,y))
            plotAndShow(0)
            print Win[idx()],'[',(len(Lst[idx()])-1),']: (',x,',',y,')' 
            cv2.setMouseCallback(Win[idx(1)], onMseClk)
        if event == cv2.EVENT_MOUSEMOVE:
            plotAndShow(1,x,y)
        
    cv2.setMouseCallback(Win[idx()], onMseClk)
    k,i=0,0
    while k not in [ord('q'),27]:
        plotAndShow(0)
        k = cv2.waitKey(0) & 0xFF
        if chr(k) == 'd':
            undo()
        elif chr(k) == 'l':
            i +=1
            Img[0] = cv2.imread(fn+"{0:0>5}".format(i)+'.jpg')
        elif chr(k) == 'r':
            i = 0 if i==0 else (i-1)
            Img[0] = cv2.imread(fn+"{0:0>5}".format(i)+'.jpg')            
        elif k in [ord('q'),27]:
            k = end(k)                
    print Lst
    cv2.destroyAllWindows()





#==============================

#specify the last Frame. Also scale can be adjusted, although 0.4 or less than 1/2 will fit better in standard screen 
def recon(fn,lastFr=8086, scale=.4):
    fn = fn.split('.')[0]
    
    H = np.array(json.load(open(fn+'.map','r')))

    try:
        os.rename(fn+'_H.avi',datetime.datetime.now().strftime("%Y%m%d-%H%M")+'H.avi')
    except:
        print 'nothing to backup'

    h,w,_ = cv2.imread(fn+"{0:0>5}".format(0)+'.jpg').shape        
    out = cv2.VideoWriter(fn+'_H.avi',cv2.VideoWriter_fourcc(*"DIVX"),28,(w,h),True)
    
    for i in range(lastFr):
        im1 = cv2.imread(fn+"{0:0>5}".format(i)+'.jpg')        
        im2 = cv2.warpPerspective(im1, H, (w,h))              
        out.write(im2)
    cv2.destroyAllWindows()
    out.release()
    print 'saving file'
    

import sys
if len(sys.argv)>=3:
    if sys.argv[1]=='decom':
        decom(sys.argv[2])
    elif sys.argv[1]=='click':
        if len(sys.argv)==3:
            click(sys.argv[2])
        elif len(sys.argv)==4 and sys.argv[3].lower()=='on':
            click(sys.argv[2],True)
    elif sys.argv[1]=='map':
        mapModel(sys.argv[2])
    elif sys.argv[1]=='recon':
        if len(sys.argv)>4:
            recon(sys.argv[2],sys.argv[3],sys.argv[4])
        elif len(sys.argv)>3:
            recon(sys.argv[2],sys.argv[3])
        elif len(sys.argv)==3:
            recon(sys.argv[2])            
    else:
        print 'Cant identify the method. Pls try again'
else:
    print 'Pls specify the method to run. Try again.'