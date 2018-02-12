#!/usr/local/lcls/package/python/current/bin/python
import numpy as np
import matplotlib.pyplot as plt
import time
import nfit
from datetime import datetime


def getPvs():

    pvs = []
    f = open('../lclsparams')
    for line in f:
        if line[0] == '#':
            pass
        else:
            pvs.append(str(line.rstrip('\n')))
    print pvs
    return pvs


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
    plt.scatter(d1,d2,c=c,s=100,lw=0.5,alpha=0.75)
    plt.xlabel(str(name1))
    plt.ylabel(str(name2))
    plt.title("2D Parameter Space")
    plt.colorbar()

def parab(data,name):

    plt.figure()
    plt.grid()
    d1 = data[name]
    c =  data['GDET:FEE1:241:ENRC']
    plt.scatter(d1,c,c=c,s=75,alpha=0.75)
    xfit = np.linspace(min(d1),max(d1),len(d1))
    plt.plot(xfit,nfit.fit(c,xfit,2)[0])
    plt.axvline(x=d1[-1])
    plt.xlabel(str(name))
    plt.ylabel('GDET:FEE1:241:ENRC')
    plt.title("Device vs GDET")


def getToday():

    ts = time.time()
    date = str(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S'))
    t = str(date[0:-9])+'/'
    return t

def plotFirstParams(pvs,keys,data):

    matches = set(pvs) & set(keys)
    print matches
    scatter(data,keys[1],keys[2])

    for k in keys:
        parab(data,k)


pvs = getPvs()
#s = '../data/'+getToday()+'lastscan.npy'
s = '../data/2016-01-10/04-01-08.npy'
print s
data = np.load(s)[-1]
keys = data.keys()
plotFirstParams(pvs,keys,data)
plt.show()
