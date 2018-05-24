#!/usr/bin/env python
# -*- coding: utf-8 -*-
#<examples/doc_basic.py>
#http://cars9.uchicago.edu/software/python/lmfit/parameters.html
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from corrplot import *
#import mpl_scatter_density # https://github.com/astrofrog/mpl-scatter-density

# function fitcorrplot

# takes a path to a CorrelationPlot scan of two PVs and the objective PV (e.g. GDET)

# loads FEL vs 2 PV data and fits a binormal variate to it

# returns the fit parameters 'amp','xm','sx','ym','sy','rho' defined in the fit function fcn2min below

def plotcorrplot(quadScanPath):

    # path to CSV with 2-quad scan data
    #quadScanPath = 'CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat';

    nquads = 2; # maybe one day we'll try 3 quad scans

    # import data to be fitted
    #[data, legend] = corrplot(quadScanPath) # read in corrplot data
    #[data, legend, ebeam_energy, photon_energy] = corrplot(quadScanPath) # read in corrplot data
    [data, legend, ebeam_energy, photon_energy, gdat80percent, gdatmax] = corrplot(quadScanPath) # read in corrplot data

    # meshgrid for the fit
    #x = y = np.linspace(-3, 3, ngrid)
    x = np.array([a for a in data[0]]); ux = np.sort(np.unique(x))
    y = np.array([a for a in data[1]]); uy = np.sort(np.unique(y))
    z = np.array([a for a in data[2]])
    #z = np.array([a for a in gdat80percent])
    #z = np.array([a for a in gdatmax])
    #dz = np.array([a for a in data[3]]);
    
    zmap = []
    for xi in ux:
        xrow = []
        for yi in uy:
            xrow += [z[(x==xi)*(y==yi)][0]]
        zmap += [xrow]
    
    #print 'len(x), len(ux) = ', len(x), ', ', len(ux)
    #print 'len(y), len(uy) = ', len(y), ', ', len(uy)
    
    extent = [min(x), max(x), min(y), max(y)] #left, right, bottom, top   
    plt.imshow(zmap, origin='lower', cmap='hot', interpolation='nearest', extent=extent)
    plt.title(legend[-1]+'   '+str(int(round(ebeam_energy)))+' MeV   '+str(int(round(photon_energy)))+' eV')
    plt.xlabel(legend[0])
    plt.ylabel(legend[1])
    plt.colorbar()
    plt.savefig('corrplot.png') # save figure
    plt.show()
    plt.close()

    ## grab peak value & locate peak
    #zpeak = np.max(z)
    #elpeak = (z == zpeak).nonzero()
    #xpeak = np.mean(x[elpeak])
    #ypeak = np.mean(y[elpeak])
    #dzpeak = np.mean(dz[elpeak])

    ## grab half-max region & find width of peak
    #elhm = (z >= 0.5*zpeak).nonzero()
    #xfwhm = np.max(x[elhm]) - np.min(x[elhm])
    #yfwhm = np.max(y[elhm]) - np.min(y[elhm])

    

