#!/usr/local/lcls/package/python/current/bin/python
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def getToday():

    ts = time.time()
    date = str(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S'))
    t = str(date[0:-9])+'/'
    return t


def simplexMovie(data,name1,name2):

    plt.figure()
    plt.grid()
    d1 = data[name1]
    d2 = data[name2]

    c =  data['GDET:FEE1:241:ENRC']
    #c =  data['timestamp'][1:]
    print c,len(d1),len(d2),len(c)
    plt.scatter(d1,d2,c=c,s=70,lw=0.5,alpha=0.75)

    coef=1
    plt.xlim([ min(d1)*coef,max(d1)*coef ])
    plt.ylim([ min(d2)*coef,max(d2)*coef ])

    sizes = [25,50,75,100]
    #markersize=sizes
    line, = plt.plot([0,0,0,0],[0,0,0,0],'bo-',lw=3,alpha=.75)
    for i in range(len(d1)-3):

        x=np.hstack((d1[i:i+3],d1[i]))
        y=np.hstack((d2[i:i+3],d2[i]))
        print x
        print y
        line.set_xdata( x )
        line.set_ydata( y )
        plt.pause(0.1)


def openAll(path):

    n1 = "QUAD:LTU1:620:BCTRL"
    n2 = "QUAD:LTU1:640:BCTRL"
    data = np.load(path+'/'+'lastscan.npy')[-1]
    print data
    simplexMovie(data,n1,n2)

    dirs = os.listdir(path)
    for i in dirs[0:3]:
        p = path+'/'+i
        data = np.load(p)[-1]
        try:
            simplexMovie(data,n1,n2)
        except:
            pass


data = np.load('./data/'+getToday()+'lastscan.npy')[-1]
#data = np.load('./data/2015-11-23/lastscan.npy')[-1]
#n1 = "QUAD:LTU1:620:BCTRL"
#n2 = "QUAD:LTU1:640:BCTRL"
#n1 = "FBCK:FB01:TR03:S1DES"
#n2 = "FBCK:FB01:TR03:S2DES"

simplexMovie(data,n1,n2)
#openAll(path)
#plt.show()
