#!/usr/local/lcls/package/python/current/bin/python
import numpy as np
import matplotlib.pyplot as plt
import extractMat
import matplotlib
import os

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

def getDataSet(d,f):
    path = d+f
    #print "PATH = ",path
    data=extractMat.extract(path,give=True)
    return data


def getDataList(filenames,dirs):

    data_sets = []
    for d in dirs:
        for f in filenames:
            try:
                name = d+f
                dset = getDataSet(d,f)
                #exit if file doesn't exist
                if type(dset) == str:
                    continue
                print "Sucsess getting",name
                data_sets.append(dset)
            except:
                print " --- ERROR getting",name
    print "Number of data sets extracted =",len(data_sets)
    return data_sets


def plotSeq(data_sets):

    to = data_sets[0]['timestamps'][0]
    for data in data_sets:
        plt.plot((data['timestamps']-to)/60.0,data["GDET_FEE1_241_ENRCHSTBR"],lw=2,alpha=0.75)



#define file names and directories
dirs      = [
                "/u1/lcls/matlab/data/2016/2016-06/2016-06-15/",
                "/u1/lcls/matlab/data/2016/2016-06/2016-06-16/",
                ]

#title = "9.5 keV simplex optimization, 4-6 device/scan "
#filenames = [
#                'OcelotScan-2016-06-15-234124.mat',
#                'OcelotScan-2016-06-15-234424.mat',
#                'OcelotScan-2016-06-15-234628.mat',
#                'OcelotScan-2016-06-15-234711.mat',
#                'OcelotScan-2016-06-15-234856.mat',
#                'OcelotScan-2016-06-15-235059.mat',
#                ]
#
#
##first gp, small device sets
#filenames = [
#                "OcelotScan-2016-06-15-235534",
#                "OcelotScan-2016-06-15-235818",
#                #"OcelotScan-2016-06-16-000917",
#                #OcelotScan-2016-06-16-001056
#               ]
#
##second gp, small device sets
#
#title = "8.2 keV GP optimization, 4-6 device/scan "
#filenames = [
#                'OcelotScan-2016-06-16-004532.mat',
#                'OcelotScan-2016-06-16-004742.mat',
#                'OcelotScan-2016-06-16-004937.mat',
#                'OcelotScan-2016-06-16-005227.mat',
#                'OcelotScan-2016-06-16-005510.mat',
#                'OcelotScan-2016-06-16-005752.mat',
#                ]

title = "8.2 keV GP optimization, 10-12 device/scan "
filenames = [
                "OcelotScan-2016-06-16-010757.mat",
                "OcelotScan-2016-06-16-011230.mat",
                ]

#get the data sets
data = getDataList(filenames,dirs)
plotSeq(data)

plt.title(title)
plt.xlabel("Time (minutes)")
plt.ylabel("Pulse Intensity (mJ)")

plt.grid()
plt.savefig('plotOut.png',format='png',dpi=300)
os.system("eog ./plotOut.png &")
#plt.show()
