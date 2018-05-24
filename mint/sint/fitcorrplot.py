#!/usr/bin/env python
# -*- coding: utf-8 -*-
#<examples/doc_basic.py>
#http://cars9.uchicago.edu/software/python/lmfit/parameters.html
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from corrplot import *

# function fitcorrplot

# takes a path to a CorrelationPlot scan of two PVs and the objective PV (e.g. GDET)

# loads FEL vs 2 PV data and fits a binormal variate to it

# returns the fit parameters 'amp','xm','sx','ym','sy','rho' defined in the fit function fcn2min below

def fitcorrplot(quadScanPath, fitNoiseQ=False):

    # path to CSV with 2-quad scan data
    #quadScanPath = 'CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat';

    nquads = 2; # maybe one day we'll try 3 quad scans

    # import data to be fitted
    #[data, legend] = corrplot(quadScanPath) # read in corrplot data
    [data, legend, ebeam_energy, photon_energy] = corrplot(quadScanPath) # read in corrplot data

    # meshgrid for the fit
    #x = y = np.linspace(-3, 3, ngrid)
    x = np.array([a for a in data[0]]);
    y = np.array([a for a in data[1]]);
    z = np.array([a for a in data[2]]);
    dz = np.array([a for a in data[3]]);

    # grab peak value & locate peak
    zpeak = np.max(z)
    elpeak = (z == zpeak).nonzero()
    xpeak = np.mean(x[elpeak])
    ypeak = np.mean(y[elpeak])
    dzpeak = np.mean(dz[elpeak])

    # grab half-max region & find width of peak
    elhm = (z >= 0.5*zpeak).nonzero()
    xfwhm = np.max(x[elhm]) - np.min(x[elhm])
    yfwhm = np.max(y[elhm]) - np.min(y[elhm])

    # in case we want to fit the map of fluctuations
    if(fitNoiseQ):
        z = dz
        dzpeak = 1.

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, y, z, dz):
        """ model decaying sine wave, subtract data"""
        amp = params['amp']
        xm = params['xm']
        sx = params['sx']
        ym = params['ym']
        sy = params['sy']
        rho = params['rho']
        if fitNoiseQ:
            bg = params['bg']
            model = bg + amp * np.exp(-0.5*((x-xm)*(x-xm)/sx/sx+(y-ym)*(y-ym)/sy/sy-2.*rho*(x-xm)*(y-ym)/sx/sy)/(1.-rho*rho))
        else:
            model = amp * np.exp(-0.5*((x-xm)*(x-xm)/sx/sx+(y-ym)*(y-ym)/sy/sy-2.*rho*(x-xm)*(y-ym)/sx/sy)/(1.-rho*rho))
        resid = (model - z) / dzpeak # replace dzpeak with dz to include some uncertainty in the fit
        #print(np.sum(resid**2))
        return resid.flatten()

    # create a set of Parameters
    params = Parameters()
    params.add('amp',   value= zpeak,  min=0., max = zpeak + 3.* dzpeak)
    params.add('xm', value= xpeak, min=xpeak-xfwhm, max=xpeak+xfwhm)
    params.add('sx', value= xfwhm/2.35, min=0.5*xfwhm/2.35, max=2.*xfwhm)
    params.add('ym', value= ypeak, min=ypeak-yfwhm, max=ypeak+yfwhm)
    params.add('sy', value= yfwhm/2.35, min=0.5*yfwhm/2.35, max=2.*yfwhm)
    params.add('rho', value= 0., min=-1., max=1.)
    if fitNoiseQ: params.add('bg', value= 0., min=-0.1, max=0.1)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, z, dz))
    kws  = {'options': {'maxiter':100}}
    result = minner.minimize()


    # calculate final result
    #final = data + result.residual

    # write error report
    report_fit(result)

    #print(result.params.items())

    #print([par for pname,par in result.params.items()])

    if fitNoiseQ:
        parnames = ['amp','xm','sx','ym','sy','rho','bg'];
    else:
        parnames = ['amp','xm','sx','ym','sy','rho'];
    parvals = [result.params.valuesdict()[p] for p in parnames];
    #print(parnames)
    #print(parvals)

    return parvals;

    # now that we have a fit, we can extrapolate
    # could make this into a function that returns an interpolated + extrapolated interface for the GP to search

#import sys

#if(len(sys.argv) > 1):
#    fitcorrplot(sys.argv[1])
