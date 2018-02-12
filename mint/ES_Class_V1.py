#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:07:24 2017

@author: Alexander Scheinker
"""

import numpy as np

class ES_min:
    #def __init__(self, pin, error_func, step, pmax, pmin, k=1, alpha=1, w0=100, alphaES=0.01):
    def __init__(self):
        k=1
        alpha=100
        w0=100
        alphaES=0.01
        
        #self.pmax = pmax
        #self.pmin = pmin
        #self.pin = pin
        #self.error_func = error_func
        #self.step = step
        self.k = k
        self.alpha = alpha
        self.alphaES = alphaES
        self.w0 = w0
        #self.nparams = len(pin)
        self.alphaES = alphaES
        #self.wES = w0*(0.5*(1+np.arange(nparams))/(nparams+0.0)+1)
        self.dtES = 2*np.pi/(20*w0)
        self.max_iter = 2
        
        
    def minimize(self, error_func, x):
        " error_func is a function of vector x and returns the cost"
        self.error_func = error_func

        " Set upper and lower bounds"
        self.pmax = x + np.abs(x)*0.1
        self.pmin = x - np.abs(x)*0.1
        self.nparams = len(x)
        
        self.pin = x
        self.wES = self.w0*(0.5*(1+np.arange(self.nparams))/(self.nparams+0.0)+1)
        
        " Use first 2 steps to get a rough understanding of sensitivity "

        " Save initial parameter values "
        self.p1ES = self.ES_normalize(x)
    
        " Save first cost "
        self.c1ES = error_func(x)
    
        " Make small changes in initial parameters and save "
        self.p2ES = self.p1ES + (self.pmax-self.pmin)/10000
    
        " Calculate unnormalized parameter value for next cost "
        pnew = self.ES_UNnormalize(self.p2ES)
        
        " Save new cost "
        self.c2ES = error_func(x)
                
        " Create kES based on initial costs and parameters and save"
        self.kES = self.k*self.ES_sensitivity(self.p1ES,self.p2ES,self.c1ES,self.c2ES)
        
        cost_val = self.c1ES
            
        
        " Now start the ES process "
        for i in range(self.max_iter):
               
            " Normalize parameters within [-1 1] "
            pnorm = self.ES_normalize(pnew)
            
            " Perform ES update "
            pnorm = pnorm + self.dtES*np.cos(self.wES*i*self.dtES+self.kES*cost_val)*(self.alpha*self.alphaES*self.wES)**0.5
            
            " Check that parameters stay within normalized range [-1, 1] "
            for jn in np.arange(self.nparams):
                if abs(pnorm[jn]) > 1:
                    pnorm[jn]=pnorm[jn]/abs(pnorm[jn])
                
            " Calculate unnormalized parameter value for next cost "
            pnew = self.ES_UNnormalize(pnorm)
                
            cost_val = error_func(pnew)
            
        return cost_val
        
    
        
    def ES_normalize(self,p):
        " Normalize parameter values to within [-1 1] "
        pdiff = (self.pmax - self.pmin)/2
        for i in range(len(pdiff)):
            if pdiff[i] == 0: pdiff[i] = 1.
        pmean = (self.pmax + self.pmin)/2
        pnorm = (p - pmean)/pdiff
        return pnorm
    
    def ES_UNnormalize(self,p):
        " Un normalize parameters back to physical values "
        pdiff = (self.pmax - self.pmin)/2
        pmean = (self.pmax + self.pmin)/2
        pUNnorm = p*pdiff + pmean
        return pUNnorm
    
    def ES_sensitivity(self,p1,p2,c1,c2):
        " Calculate total change in cost relative to change in parameters "
        dcdp = max(abs(sum((c2-c1)/(p2-p1))),0.1)
        if dcdp > 0:
            self.kES = 2*(self.w0/self.alphaES)**0.5/dcdp
        else: 
            self.kES = 1
        return self.kES
