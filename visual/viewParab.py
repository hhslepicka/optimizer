#!/usr/local/lcls/package/python/current/bin/python
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def scatter(data,name1,name2):

    plt.figure()
    plt.grid()
    d1 = data[name1]
    d2 = data[name2]
    c =  data['GDET:FEE1:241:ENRC']
    #c =  data['timestamp'][1:]
    print "LENGTHS"
    print len(d1)
    print len(d2)
    print len(c)
    print
    #plt.scatter(d1,d2,c=c,s=100,lw=0.5,alpha=0.75)
    #plt.scatter(d1,d2,c=c,s=100,lw=0.5,alpha=0.75)
    plt.plot(d1,c,'bo',alpha=0.7)
    plt.xlabel(str(name1))
    plt.ylabel('GDET:FEE1:241:ENRC')
    plt.title("parameter space")
    #plt.colorbar()

def getToday():

    ts = time.time()
    date = str(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S'))
    t = str(date[0:-9])+'/'
    return t


data = np.load('./data/'+getToday()+'lastscan.npy')[-1]
#n1 = "FBCK:FB01:TR03:S1DES"
#n2 = "FBCK:FB01:TR03:S2DES"
n1 = "QUAD:LTU1:620:BCTRL"
n2 = "QUAD:LTU1:660:BCTRL"
#n1 = 'YCOR:IN20:952:BCTRL'
#n2 = 'XCOR:IN20:951:BCTRL'
#n1 = "QUAD:LI26:601:BCTRL"
#n2 = "QUAD:LI26:701:BCTRL"
scatter(data,n1,n2)
plt.show()
