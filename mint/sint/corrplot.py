# -*- coding: utf-8 -*-
# 2017-09-24 added energy retrieval for BEND_DMP1_400_BDES

import numpy as np
import pandas as pd
import string
import os
import scipy.io as sio

def corrplot(fname):

    fcnname = "corrplot"

    goalPV = "GDET:FEE1:241:ENRC"
    energyPV = 'BEND:DMP1:400:BDES'

    #fname = "CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat";
    #print (fcnname + " - warning: overriding input filename with " + fname + " for testing");

    try:
        dat = sio.loadmat(fname)
    except:
        print(fcnname + " - warning: Bad file")
        return 'Bad file'

    data = dat['data']

    # how many control PVs?
    nctrl = len(data['ctrlPV'][0][0])
    
    # how many read PVs?
    nread = len(data['readPV'][0][0])

    # how many samples?
    nsamp = len(data['ctrlPV'][0][0][0])

    # get names
    ctrlPVs = np.array([data['ctrlPV'][0][0][i][0][0][0] for i in range(nctrl)]);
    readPVs = np.array([data['readPV'][0][0][i][0][0][0] for i in range(nread)]);

    # get data
    cdat = np.array([[b[1][0][0] for b in a] for a in data['ctrlPV'][0][0]]);
    rdat = np.array([[b[1][0][0] for b in a] for a in data['readPV'][0][0]]);

    # energy PV index
    try:
        eepvi = np.arange(nread)[np.array([(name == 'BLD:SYS0:500:ENERGY')[0] for name in readPVs])][0]
        eedat = np.array([np.median([b[1] for b in a]) for a in data['readPV'][0][0][eepvi]]);
        eedat = eedat[~np.isnan(eedat)] # remove nans
        ebeam_energy = np.median(eedat)
    except:
        ebeam_energy = 7
    
    # photon energy PV index
    try:
        pepvi = np.arange(nread)[np.array([(name == 'BLD:SYS0:500:PHOTONENERGY')[0] for name in readPVs])][0]
        pedat = np.array([np.median([b[1] for b in a]) for a in data['readPV'][0][0][pepvi]]);
        pedat = pedat[~np.isnan(pedat)] # remove nans
        photon_energy = np.median(pedat)
    except:
        photon_energy = 7

    # find goal PV index
    # gpvi = 1;
    gpvi = np.arange(nread)[np.array([(name == 'GDET:FEE1:241:ENRC')[0] for name in readPVs])][0]

    # get median gdet data for each point
    gdat = np.array([np.median([b[1] for b in a]) for a in data['readPV'][0][0][gpvi]]);

    def meddev(nparray):
        med = np.median(nparray);
        lst = np.abs(nparray - med);
        return np.median(lst);

    # get median deviation gdet data for each point
    gdatmeddev = np.array([meddev([b[1] for b in a]) for a in data['readPV'][0][0][1]]);
    gdatmeddev = 1.4826 * gdatmeddev; # scale median deviation to equal rms for a gaussian

    # sanity check
    if( len(gdat) != nsamp ):
        print(fcnname + " - error: have more " + goalPV + " points than ctrlPV points");

    # assemble the data array
    #adat = np.reshape(np.append(cdat, gdat),(nctrl+1,nsamp))
    adat = np.reshape(np.append(np.append(cdat, gdat), gdatmeddev),(nctrl+2,nsamp))

    # assemble the header
    legend = np.append(ctrlPVs,goalPV)

    # return legend and data
    return [adat, legend, ebeam_energy, photon_energy]
    #return adat



def corrplot_all(fname):

    fcnname = "corrplot_all";

    goalPV = "GDET:FEE1:241:ENRC";

    #fname = "CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat";
    #print (fcnname + " - warning: overriding input filename with " + fname + " for testing");


    try:
        dat = sio.loadmat(fname)
    except:
        print(fcnname + " - warning: Bad file");
        return 'Bad file'

    data = dat['data']

    # how many control PVs?
    nctrl = len(data['ctrlPV'][0][0])

    # how many samples?corrplot2csv
    nsamp = len(data['ctrlPV'][0][0][0])

    # get names
    ctrlPVs = np.array([data['ctrlPV'][0][0][i][0][0][0] for i in range(nctrl)]);

    # get ctrlPV data
    cdat = np.array([[b[1][0][0] for b in a] for a in data['ctrlPV'][0][0]]);

    # find goal PV index
    gpvi = 1; # <------------------------- todo

    # get median gdet data for each point
    gdat = np.array([np.median([b[1] for b in a]) for a in data['readPV'][0][0][1]]);

    def meddev(nparray):
        med = np.median(nparray);
        lst = np.abs(nparray - med);
        return np.median(lst);

    # get median deviation gdet data for each point
    gdat
    gdatmeddev = np.array([meddev([b[1] for b in a]) for a in data['readPV'][0][0][1]]);
    gdatmeddev = 1.4826 * gdatmeddev; # scale median deviation to equal rms for a gaussian

    # sanity check
    if( len(gdat) != nsamp ):
        print(fcnname + " - error: have more " + goalPV + " points than ctrlPV points");

    # assemble the data array
    #adat = np.reshape(np.append(cdat, gdat),(nctrl+1,nsamp))
    adat = np.reshape(np.append(np.append(cdat, gdat), gdatmeddev),(nctrl+2,nsamp))

    # assemble the header
    legend = np.append(ctrlPVs,goalPV)

    # return legend and data
    return [adat, legend]
    #return adat
