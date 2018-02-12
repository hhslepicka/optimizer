# -*- coding: utf-8 -*-

"""
simulation interface

based on LCLSMachineInterface - Machine interface file for the LCLS to ocelot optimizer

-----------------------

Contains simple interfaces for the Bayes optimization class.

from Mitch: The reason that interface getState() methods return both x and y is to avoid making the assumption that the optimizer is the only thing that has access to the machine/interface. As it is written currently, the optimizer is robust to someone else tweaking quads while it is working (although this is probably still not a good idea).

Each interface must have the getState and setX methods as used below.
"""

import numpy as np
from numpy.random import rand
import time
import sys
import math
from scipy.special import gamma
from scipy.special import erfinv


#____________________________________________________________________
# an interface that samples from a multivariate normal distribution
#class MultinormalInterface(object):
class MultinormalInterface:
    """ Start acquisition interface class """

    #_____________________________________________
    def __init__(self, offsets, projected_widths, correlation_matrix):
        # interface name
        self.name = 'MultinormalInterface'
        
        # default statistic to tune on
        self.stat_name = 'Mean'
        self.ebeam_energy = 7. # GeV

        """ Initialize parameters for the scanner class. """
        self.secs_to_ave = 2         #time to integrate gas detector

        self.points = 1
        self.nsamples = self.points;
        self.simmode = 1 # 0: multinormal distribution
                         # 1: correlation plot fit
        self.quickNoiseQ = False # if True: only one sample is drawn with uncert = standard error of the mean

        # making this its own function in case we want to call again later
        self.store_moments(offsets, projected_widths, correlation_matrix)

        self.y = -1
        self.mean = 0
        self.stdev = 0
        self.stdev_nsample = 0

        self.sigAmp = 1.
        self.bgNoise = 0.064 # something typical
        self.sigNoiseScaleFactor = 0.109 # seems like something typical is amp_noise / sqrt(amp_signal) ~= 0.193/np.sqrt(3.113) = 0.109
        self.noiseScaleFactor = 1. # easy to use this as a noise toggle

        self.numSamples = 1.
        self.numBatchSamples = 1.
        self.SNRgoal = 0 # specify SNR goal; if 0, then numSamples is unchanged
        self.maxNumSamples = 30.*120. # a limit on how long people would want to wait if changing numSamples to achieve the SNRgoal

        self.last_numSamples = self.numSamples
        self.last_SNR = self.SNRgoal

        #if(self.simmode == 1 or True):
        self.getter = self     #getter class for channel access
        #self.pvs = self.pvs # pv names stored here
        #self.detector = self.pvs[-1]

        ## reference for goal
        #self.pvs_optimum_value = np.array([self.mean_xm, self.mean_ym, self.mean_amp])
        #self.detector_optimum_value = self.pvs_optimum_value[-1]


    ############################################################
    # main public member functions


#    #_____________________________________________
#    # Estimate covariance hyperparameters
#    def estCovarParams(self):
#        return np.log(0.5/(self.sigmas**2)
#
#    #_____________________________________________
#    # Estimate amplitude hyperparameters
#    def estAmpParam(self):
#        return 2.*np.log(self.sigAmp/np.sqrt(2.*np.pi))
#
#    #_____________________________________________
#    # Estimate noise hyperparameters
#    def estNoiseParam(self):
#        noise = np.abs(self.bgNoise) + np.abs(self.sigNoiseScaleFactor) * np.sqrt(self.sigAmp)
#        noise = noise * (self.noiseScaleFactor+1.e-6) / np.sqrt(self.points)
#        return 2.*np.log(noise)

    #_____________________________________________
    # The UI requests how much time we need to trim magnets
    def dataDelay(self, objective_func_pv, numPulse):
        self.points = numPulse
        self.nsamples = self.points;
        return 0.000001

    #_____________________________________________
    # Something LCLS specific that should probably be contained in LCLSMachineInterface.__init__
    def setListener(self, anyNumber):
        pass

    #_____________________________________________
    # Something LCLS specific that should probably be contained in LCLSMachineInterface.__init__
    def get_charge_current(self):
        charge = 'None'
        current = 'None'
        return charge, current

    #_____________________________________________
    # Get fcn takes 1 pv string or a list of pvs
    # In a real machine setting, this fcn should check the machine status before returning values
    def get_value(self, variable_names):

        # if a list of names, return a list of values
        #if hasattr(variable_names, '__iter__'):
        #    print 'getting a value', self.get1(var)
        #    return np.array([self.get1(var) for var in variable_names])
        #return self.get1(var)
        # if one variable, return one value
        #else:
        return self.get1(variable_names)

    #_____________________________________________
    # Get fcn takes 1 pv string or a list of pvs
    # In a real machine setting, this fcn should check the machine status and wait for devices to settle before returning values
    def set_value(self, variable_names, values):

        # if a list of names, return a list of values
        #if hasattr(variable_names, '__iter__') and hasattr(values, '__iter__'):
        #    if len(variable_names) == len(values):
        #        for i in range(len(variable_names)):
        #            self.set1(variable_names[i], values[i])
        #    else:
        #        print "SimulationInterface.get - ERROR: Inputs must have same number of entries."
        #
        #elif hasattr(variable_names, '__iter__') or hasattr(values, '__iter__'):
        #    print "SimulationInterface.get - ERROR: Inputs must both be listable."

        # if one variable, set one value
        #else:
        #    print 'setting a value', values
        self.set1(variable_names, values)

    #def initErrorCheck(self): # not needed (call errorCheck or something or do this in the setup from the constructor
        #"""
        #Initialize PVs and setting used in the errorCheck method.
        #"""
        #return

    def get(self, pv):
        #return self.get_value(pv)
        print 'getting', pv, self.get1(pv)
        return self.get1(pv)

    def get_energy(self):
        return self.ebeam_energy

    def put(self, pv, val):
        print 'setting', pv, self.set1(pv, val)

    def errorCheck(self):
    #def get_status(self): # how is errorCheck used? maybe you want a get_status fcn instead: OK, WAIT, BREAK
        """
        Method that check the state of BCS, MPS, Gaurdian, UND-TMIT and pauses GP if there is a problem.
        """
        return

    #def get_sase(self):
    #def get_objective(self, nsamples = 100, returnErrorQ = False): # where is it best to set this up?
    def get_sase(self, data, points = None, returnErrorQ = False):
        """
        Returns an average for the objective function from the selected detector PV.

        Args:
        nsamples (int): integer number of samples to average over
        returnErrorQ (bool): option to return stdev too

        Returns:
        Float of detector measurment mean (and optionally standard deviation)

        Ideas:  - option to do median & median deviation
                - perhaps we'd prefer to specify a precision tolerance? i.e. sample until abs(mean(data[-n:]) - mean(data[-(n-1):])) / std(data[-n:]) < tolerance  or  abs(data[-n] / n) / std(data) < tolerance

        """

        # update number of points to sample
        if(points != self.points):
            self.points = points
            self.nsamples = self.points


        # acquire data
        #data = np.array([self.get_value(self.detector) for i in range(nsamples)])

        ## perform stats
            #try:
                #datamean = np.mean(data)
                #datadev  = np.std(data)
            #except:
                #datamean = data
                #datadev = data
            #print datamean, datadev

            ## record stats
        ##self.record_data(datamean,datadev) # do we want the ctrl interface to record?

        ## return stats
        #if returnErrorQ:
                #return np.array([datamean, datadev])
            #else:
                #return datamean, datadev

        if self.quickNoiseQ is True:

            # NOTE: self.stdev is standard deviation for 1 sample
            #       self.stderr_of_mean is the standard error of the mean

            data = self.f(self.x)

            print "objective mean, stderr_of_mean, and snr for " + str(self.points) + " acquired points is " + str([np.mean(data), self.stderr_of_mean, np.mean(data)/self.stderr_of_mean])

            #print "self.stdev[0,0] = ",self.stdev[0,0],"\tself.bgNoise = ",self.bgNoise,"\nself.stdev[0,0]/self.bgNoise*np.sqrt(self.points) = ",self.stdev[0,0]/self.bgNoise*np.sqrt(self.points)
            #print "np.std(data) = ", np.std(data),"/ndata = ",data

            #return np.mean(data), self.stdev
            
            # sample standard deviation and standard error of the sample stdev
            #print 'gamma(points/2.) = ',gamma(points/2.),'\ngamma((points-1.)/2.) = ',gamma((points-1.)/2.)
            c4n = np.sqrt(2./(points-1.))*gamma(points/2.)/gamma((points-1.)/2.)
            stdev_stderr = self.stdev[0,0] * np.sqrt(c4n**-2 - 1.)
            sample_stdev = self.stdev[0,0]+np.random.randn()*stdev_stderr
            stderr_of_mean = max([sample_stdev / np.sqrt(points), 1.e-10])
            #print 'points = ',points,'\nc4n = ',c4n,'\nstdev_stderr = ',stdev_stderr,'sample_stdev = ',sample_stdev
            #return sample_stdev / self.bgNoise - 1., sample_stdev # better suppression


            if self.stat_name == 'Mean' or self.stat_name == 'Median':
                return np.mean(data),stderr_of_mean
            elif self.stat_name == '80th percentile':
                # percentile cut
                pvalue = 0.8 # pvalue = percentile/100.
                zscore = np.sqrt(2) * erfinv(2*pvalue-1)
                return zscore * sample_stdev, stderr_of_mean
            elif self.stat_name == '20th percentile':
                # percentile cut
                pvalue = 0.2 # pvalue = percentile/100.
                zscore = np.sqrt(2) * erfinv(2*pvalue-1)
                return zscore * sample_stdev, stderr_of_mean
            elif self.stat_name == 'Standard deviation':
                return self.sample_stdev, stdev_stderr
            else:
                print "MultinormalInterface - WARNING: not sampling so ignoring gui set statistic for now. Returning mean"
                return np.mean(data),stderr_of_mean

        else:
            
            #data = np.array([self.f(self.x) for i in range(int(points))])
            data = self.f(self.x)
            #data = self.f(self.x)

            #print "points = ", points, "\nself.points = ",self.points,"\ndata = ", data
            #print "data = ", data
            #print "data.shape = ",data.shape,"\tdata.size = ",data.size,"\tnp.std(data) = ",np.std(data)

            #return np.mean(data), np.std(data)
            #return np.std(data) / self.bgNoise - 1, np.std(data)
            #return np.max(data), np.std(data)
            #return np.mean(data[data>np.mean(data)]), np.std(data)
            #return np.percentile(data, 90), np.std(data)
            
            if self.stat_name == 'Median':
                statistic = np.median(data)
            elif self.stat_name == 'Standard deviation':
                statistic = np.std(data)
            elif self.stat_name == 'Median deviation':
                median = np.median(data)
                statistic = np.median(np.abs(data-median))
            elif self.stat_name == 'Max':
                statistic = np.max(data)
            elif self.stat_name == 'Min':
                statistic = np.min(data)
            elif self.stat_name == '80th percentile':
                statistic = np.percentile(data,80)
            elif self.stat_name == 'average of points > mean':
                dat_last = data
                percentile = np.percentile(data,50)
                statistic = np.mean(dat_last[dat_last>percentile])
            elif self.stat_name == '20th percentile':
                statistic = np.percentile(data,20)
            else:
                self.stat_name = 'Mean'
                statistic = np.mean(data)
            # check if this is even used:
            #sigma   = np.std(data) 
            sigma   = np.std(data) / np.sqrt(data.size) # standard error of the mean

            print self.stat_name, ' of ', data.size, ' points is ', statistic, ' and standard deviation is ', sigma
            
            return statistic, sigma
            

    def get_limits(self, device,percent=0.25):
        """
        Function to get device limits.
        Executes on every iteration of the optimizer function evaluation.
        Currently does not work with the normalization scheme.
        Defaults to + 25% of the devices current values.

        Args:
            device (str): PV name of the device to get a limit for
            percent (float): Generates a limit based on the percent away from the devices current value
        """
#        #val = self.start_values[device]
#        val = self.get(device)
#        tol = abs(val*percent)
#        lim_lo = val-tol
#        lim_hi = val+tol
#        limits = [lim_lo,lim_hi]
#        #print device, 'LIMITS ->',limits
#        return limits

        #Dosnt work with normalizaiton, big limits
        return [-1,1] # looks like this function doesn't do anything


    # looks redundant
    #def get_start_values(self,devices,percent=0.25):
        #"""
        #Function to initialize the starting values for get_limits methomethodd.

        #Called from tuning file or GUI

        #Args:
            #devices ([str]): PV list of devices
            #percent (float): Percent around the mean to generate limits

        #"""
        #self.start_values={}
        #self.norm_minmax={}
        #for d in devices:
            #val = self.getter.caget(str(d))
            #self.start_values[str(d)] = val
            #tol = abs(val*percent)
            #lim_lo = val-tol
            #lim_hi = val+tol
            #limits = [lim_lo,lim_hi]
            #self.norm_minmax[str(d)] = [lim_lo,lim_hi]

    # END LCLSMachineInterface specific functions
    # ---------------------------------

    # simple access fcn
    def get1(self, pvname):

        index = self.pvdict[pvname]
        if index == len(self.pvs)-1:
            self.getState()
            if hasattr(self.y, '__iter__'):
                return self.y[0]
            else:
                return self.y
        else:
            return self.x[-1,index]

    # simple access fcn
    def set1(self, pvname, value):
        index = self.pvdict[pvname]
        if index == len(self.pvs)-1:
            self.y = value
        else:
            self.x[-1,index] = value

    def store_moments(self, offsets, projected_widths, correlation_matrix):

        # check sizes
        if offsets.size != projected_widths.size or offsets.size != np.sqrt(correlation_matrix.size):
            print "MultinormalInterface - ERROR: Dimensions of input parameters are inconsistant."

        # store inputs
        self.offsets = offsets # list of peak location coords
        self.sigmas = np.abs(projected_widths) # list of peak widths projected to each axis
        self.corrmat = correlation_matrix # correlation matrix of peaks
        self.covarmat = np.dot(np.diag(self.sigmas),np.dot(self.corrmat,np.diag(self.sigmas))) # matrix of covariances
        self.invcovarmat = np.linalg.inv(self.covarmat) # inverse of covariance matrix (computed once and stored)

        # seems like in fint, he wants to store the last random number generated by the last fcn call so let's store some random number
        self.x = np.array(np.zeros(self.offsets.size),ndmin=2)

        # reference for goal
        self.pvs_optimum_value = np.array([self.offsets, 1.])
        self.detector_optimum_value = self.pvs_optimum_value[-1]

        # name these PVs
        self.pvs = np.array(["sim_device_" + str(i) for i in np.array(range(self.offsets.size))+1])
        self.detector = "sim_objective"
        self.pvs = np.append(self.pvs, self.detector)
        self.pvdict = dict() # for simple lookup
        for i in range(len(self.pvs)):
            self.pvdict[self.pvs[i]] = i # objective fcn is last here

#        print "\nMultinormalInterface: INFO - PVs setup:"
#        for pv in self.pvs: print(pv)


    def fmean(self, x_new): # to calculate ground truth

        self.x = x_new
        self.dx = self.x - self.offsets

        self.mean = self.sigAmp * np.exp(-0.5*np.dot(self.dx,np.dot(self.invcovarmat,self.dx.T)))

        return self.mean


    def f(self, x_new): # let this method update means and stdevs

        self.x = x_new
        self.dx = self.x - self.offsets

        # set result mean (perturb by noise below)
        self.mean = abs(self.sigAmp * np.exp(-0.5*np.dot(self.dx,np.dot(self.invcovarmat,self.dx.T))))

        # set resulting 1 sample noise (nsample noise below)
        self.stdev = np.abs(self.bgNoise) + np.abs(self.sigNoiseScaleFactor) * np.sqrt(self.mean)
        self.stdev = abs(self.noiseScaleFactor) * self.stdev

        # figure out the number of samples needed to achieve the SNRgoal
        if(self.SNRgoal > 0 and self.noiseScaleFactor > 0):

            # analytic way
            self.numSamples = min([(self.SNRgoal * self.stdev / self.mean) ** 2., self.maxNumSamples])
            self.numSamples = max([np.ceil(self.numSamples / self.numBatchSamples), 1.]) * self.numBatchSamples
            self.points = self.numSamples # for compatibility with machine interface api

            # iterative way
            #self.last_numSamples = 0
            #self.last_SNR = 0
            #while(self.last_SNR < self.SNRgoal and self.last_numSamples < self.maxNumSamples):
                #self.last_numSamples += self.numBatchSamples
                #self.noise = oneSampleNoise / np.sqrt(self.last_numSamples)
                #self.last_SNR = self.mean / self.noise

        if self.quickNoiseQ is True:
        
            # set noise
            self.stderr_of_mean = max([self.stdev / np.sqrt(self.points), 1.e-10]); # small number needed to prevent requesting stdev of 0 in np.random.normal if self.noiseScaleFactor is zero

            # perturb mean by nsample noise
            self.y = self.mean + np.random.normal(0., self.stderr_of_mean, self.mean.shape)
            #self.y = self.y + np.random.normal(0., self.stdev_nsample, self.y.shape) * (self.y > 0) # zero noise where the mean is negative

        else:

            # perturb mean by nsample noise
            self.y = np.array([self.mean + np.random.normal(0., self.stdev, self.mean.shape) for i in range(int(self.points))])

        return np.array(self.y, ndmin=2)

    def SNR(self):
        return self.mean / self.stdev_nsample

    def getState(self): # see note at top regarding
        return np.array(self.x, ndmin = 2), self.f(self.x)

    def setX(self, x_new):
        self.x = x_new



    #=======================================================#
    # ------------------- Data Saving --------------------- #
    #=======================================================#

    def recordData(self, objective_func_pv, objective_func, devices):
        """
        Get data for devices everytime the SASE is measured to save syncronous data.

        Args:
                gdet (str): String of the detector PV, usually gas detector
                simga (float): Float of the measurement standard deviation

        """
        try:
            self.data
        except:
            self.data = {} #dict of all devices deing scanned
        self.data[objective_func_pv] = [] #detector data array
        self.data['DetectorStd'] = [] #detector std array
        self.data['timestamps']  = [] #timestamp array
        self.data['charge']=[]
        self.data['current'] =[]
        self.data['stat_name'] =[]
        try:
            self.pv_list
        except:
            self.pv_list = []
        for dev in devices:
            self.data[dev.eid] = []
        #print('obj times', objective_func.times)
        for dev in devices:
            self.pv_list.append(dev.eid)
            vals = len(dev.values)
            self.data[dev.eid].append(dev.values)
        if vals<len(objective_func.values):
            objective_func.values = objective_func.values[1:]
            objective_func.times = objective_func.times[1:]
            objective_func.std_dev = objective_func.std_dev[1:]
            objective_func.charge = objective_func.charge[1:]
            objective_func.current = objective_func.current[1:]
        self.data[objective_func_pv].append(objective_func.values)
        self.data['DetectorStd'].append(objective_func.std_dev)
        self.data['timestamps'].append(objective_func.times)
        self.data['charge'].append(objective_func.charge)
        self.data['current'].append(objective_func.current)
        self.data['stat_name'].append(self.stat_name)
        return self.data

    def saveData(self, objective_func_pv, objective_func, devices, name_opt, norm_amp_coeff):
        """
        Save scan data to the physics matlab data directory.

        Uses module matlog to save data dict in machine interface file.
        """
        data_new = self.recordData(objective_func_pv, objective_func, devices)
        #get the first and last points for GDET gain
        self.detValStart = data_new[objective_func_pv][0]
        self.detValStop  = data_new[objective_func_pv][-1]
        
        #replace with matlab friendly strings
        for key in data_new:
            key2 = key.replace(":","_")
            data_new[key2] = data_new.pop(key)

        #extra into to add into the save file
        #data_new["BEND_DMP1_400_BDES"]   = self.get("BEND:DMP1:400:BDES")
        data_new["BEND_DMP1_400_BDES"]   = self.get_energy()
        data_new["Energy"]   = self.get_energy()
        data_new["ScanAlgorithm"]        = str(name_opt)      #string of the algorithm name
        data_new["ObjFuncPv"]            = str(objective_func_pv) #string identifing obj func pv
        data_new["NormAmpCoeff"]         = norm_amp_coeff
        data_new["pv_list"]              = self.pv_list

        #save data
        import simlog
        self.last_filename=simlog.save("OcelotScan",data_new,path='default')#self.save_path)

