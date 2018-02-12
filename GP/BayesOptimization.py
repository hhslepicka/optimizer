# -*- coding: utf-8 -*-
"""
Contains the Bayes optimization class.
Initialization parameters:
    model: an object with methods 'predict', 'fit', and 'update'
    interface: an object which supplies the state of the system and
        allows for changing the system's x-value.
        Should have methods '(x,y) = intfc.getState()' and 'intfc.setX(x_new)'.
        Note that this interface system is rough, and used for testing and
            as a placeholder for the machine interface.
    acq_func: specifies how the optimizer should choose its next point.
        'PI': uses probability of improvement. The interface should supply y-values.
        'EI': uses expected improvement. The interface should supply y-values.
        'UCB': uses GP upper confidence bound. No y-values needed.
        'testEI': uses EI over a finite set of points. This set must be
            provided as alt_param, and the interface need not supply
            meaningful y-values.
    xi: exploration parameter suggested in some Bayesian opt. literature
    alt_param: currently only used when acq_func=='testEI'
    m: the maximum size of model; can be ignored unless passing an untrained
        SPGP or other model which doesn't already know its own size
    bounds: a tuple of (min,max) tuples specifying search bounds for each
        input dimension. Generally leads to better performance.
        Has a different interpretation when iter_bounds is True.
    iter_bounds: if True, bounds the distance that can be moved in a single
        iteration in terms of the length scale in each dimension. Uses the
        bounds variable as a multiple of the length scales, so bounds==2
        with iter_bounds==True limits movement per iteration to two length
        scales in each dimension. Generally a good idea for safety, etc.
    prior_data: input data to train the model on initially. For convenience,
        since the model can be trained externally as well.
        Assumed to be a pandas DataFrame of shape (n, dim+1) where the last
            column contains y-values.
Methods:
    acquire(): Returns the point that maximizes the acquisition function.
        For 'testEI', returns the index of the point instead.
        For normal acquisition, currently uses the bounded L-BFGS optimizer.
            Haven't tested alternatives much.
    best_seen(): Uses the model to make predictions at every observed point,
        returning the best-performing (x,y) pair. This is more robust to noise
        than returning the best observation, but could be replaced by other,
        faster methods.
    OptIter(): The main method for Bayesian optimization. Maximizes the
        acquisition function, then uses the interface to test this point and
        update the model.
"""

import operator as op
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
#try:
    #from scipy.optimize import basinhopping
    #from parallelbasinhopping import *
    #basinhoppingQ = True
#except:
    #basinhoppingQ = False
    #pass
basinhoppingQ = False
from parallelstuff import *
import time
#import math
from copy import deepcopy
import pandas as pd

from heatmap import plotheatmap

# TODO callbacks or real-time acquisition needed: the minimizer for the acquisition fcn only looks for number of devices when loaded; not when devices change


def normVector(nparray):
    return nparray / np.linalg.norm(nparray)

class BayesOpt:
    def __init__(self, model, target_func, acq_func='EI', xi=0.0, alt_param=-1, m=200, bounds=None, iter_bound=False, prior_data=None, start_dev_vals=None, dev_ids=None, energy=None, hyper_file=None):
        self.model = model
        self.m = m
        self.bounds = bounds
        self.iter_bound = iter_bound
        self.target_func = target_func
        self.mi = self.target_func.mi
        print 'Using ', self.mi.name #LCLSMachineInterface, CorrplotInterface, #MultinormalInterface
        self.acq_func = (acq_func, xi, alt_param)
        self.max_iter = 100
        self.check = None
        self.alpha = 1
        self.kill = False
        self.ndim = np.array(start_dev_vals).size
        self.multiprocessingQ = True # speed up aquisition function optimization
        print "Bayesian optimizer set to use ", acq_func, " acquisition function"

        # DELETE AFTER PUSHING mint.GaussProcess.preprocess stuff into here
        if hyper_file == None:
            print 'BayesOpt - WARNING: hyper_file = ', hyper_file

        self.energy = energy
        self.dev_ids = dev_ids
        print 'FIX ME?! pass mi instead of dev_ids'
        self.start_dev_vals = start_dev_vals
        
        self.usePriorMean = True
        if self.usePriorMean:
            self.build_prior_mean()
        
            
    def build_prior_mean(self):

        
        ## calculate prior parameters from the seed data
        #calcPriorFromSeedData = True
        #if calcPriorFromSeedData:
            ##try:
                ### estimate from the seed data
                ##df = prior_data[prior_data.iloc[:,-1]==prior_data.iloc[:,-1].max()] # point w/ max objective fcn
                ##prcentroid = np.array(df.iloc[:,:-1])[0] # coords for center
                ##pramp = 0.2*np.array(df.iloc[:,-1])[0] # use max seen obj fcn for peak of prior
                ### IDEA: fit gaussian prior to available points for each acquisition?
            ##except:
                ### estimate from starting data
                ##prcentroid = start_dev_vals
                ##pramp = 0.01

            #try:
                ## initialize model on prior data
                #print "prior_data = ",prior_data
                #if(prior_data is not None):
                    #p_X = prior_data.iloc[:, :-1]
                    #p_Y = prior_data.iloc[:, -1]
                    #num = len(prior_data.index)
                    #self.model.fit(p_X, p_Y, min(m, num))
            #except:
                #pass

        # prior mean function definition (perhaps push unique copy into each interface
        self.model.prmean_name = 'multinormal_priorfcn(x,[xpeak,dxpeak,peakFEL]) =  peakFEL*np.exp(-0.5*(np.linalg.norm((x-xpeak)/dxpeak)**2.))'
        def multinormal_priorfcn(x,params):
            [xpeak,dxpeak,peakFEL] = params
            relfracdist = np.linalg.norm((x-xpeak)/dxpeak)
            return peakFEL*np.exp(-0.5*(relfracdist**2.)) # Mitch might have flipped the sign on the 
        
        # stuff for multinormal simulation interface
        if self.mi.name == 'MultinormalInterface':
            # hyperparams for multinormal simulation interface
            covar_params = np.array(np.log(0.5/(self.mi.sigmas**2)),ndmin=2)
            noise_param = 2.*np.log((self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15) / np.sqrt(self.mi.numSamples))
            amp_param = np.log(self.mi.sigAmp/2.)
            hyperparams = (covar_params, amp_param, noise_param)
            print "BO: changing hyperparams for multinormal sim ", hyperparams
            self.model.covar_params = hyperparams[:2]
            self.model.noise_var = np.exp(hyperparams[2])

            # prior parameters for multinormal simulation interface
            #self.model.prmeanp = [startingpoint, 0.1*startingpoint, 1.] # naive prior
            #self.model.prmeanp = [startingpoint, 1.5*np.ones(ndim), 0.1]
            lengthscales = np.sqrt(0.5*np.exp(-self.model.covar_params[0]))
            prwidths = 2.*lengthscales
            ndim = lengthscales.size

            # grab centroid from offsets in simulation mode interface
            prcentroid = np.array(self.mi.offsets,ndmin=2)

        ## stuff for CorrplotInterface simulation interface; should generally just use the scraped data params
        #overrideCorrplotPrior = False
        #if self.mi.name == 'CorrplotInterface' and overrideCorrplotPrior:
            #try: # prior mean centroid from corrplor sim interface
                #prcentroid = np.array(self.target_func.mi.pvs_optimum_value[0],ndmin=2)
            #except:
                #pass

        # kick prior mean centroid only in simulation modes
        usePriorKicks = True
        #if usePriorKicks and (self.mi.name == 'MultinormalInterface' or self.mi.name == 'CorrplotInterface'): # perturb centroid
        if usePriorKicks and (self.mi.name == 'MultinormalInterface'): # perturb centroid
            kick_nsigma = 1.#np.sqrt(ndim) # scales the magnitude of the distance between start and goal so that the distance has a zscore of nsigma
            kicks = np.random.randn(ndim) #1.*np.ones(ndim) # peak location is an array
            kicks = np.round(kicks*kick_nsigma/np.linalg.norm(kicks),2) #1.*np.ones(ndim) # peak location is an array
            kicks = kicks * lengthscales
            prcentroid = prcentroid + kicks
            print 'BayesOpt - WARNING: using prior kicks with nsigma = ',kick_nsigma

        # new data reduction prior
        if self.mi.name != 'MultinormalInterface':
            self.prior_params_file = 'parameters/fit_params_august.pkl'
            self.prior_params_file_older = 'parameters/fit_params.pkl'
            filedata_recent = pd.read_pickle(self.prior_params_file) # recent fits
            filedata_older = pd.read_pickle(self.prior_params_file_older) # fill in sparsely scanned quads with more data from larger time range
            filedata = filedata_older
            names_recent = filedata_recent.T.keys()
            names_older = filedata_older.T.keys()
            pvs=self.dev_ids
            pvs = [pv.replace(":","_") for pv in pvs]

            # load in moments for prior mean from the data fits pickle
            prcentroid = np.array(self.start_dev_vals) * 0
            prwidths = np.array(self.start_dev_vals) * 0
            self.model.prior_pv_info = [['i','pv','prior file','number of points fit for pv','prcentroid[i]','prwidths[i]','ave_m','ave_b','width_m','width_b','ave_res_m','ave_res_b','width_res_m','width_res_b']]
            for i, pv in enumerate(pvs):
                # note: we pull data from most recent runs, but to fill in the gaps, we can use data from a larger time window
                #       it seems like the best configs change with time so we prefer recent data
                
                pvprlog = [i,pv]

                # use recent data unless too sparse (less than 10 points)
                if pv in names_recent and filedata.get_value(pv, 'number of points fitted')>10:
                    print('PRIOR STUFF: ' + pv + ' RECENT DATA LOOKS GOOD')
                    filedata = filedata_recent
                    pvprlog += [self.prior_params_file] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]
                    
                # fill in for sparse data with data from a larger time range
                #elif pv in names_recent and filedata.get_value(pv, 'number of points fitted')<=10:
                elif pv in names_older:
                    print('PRIOR STUFF: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_older
                    self.prior_params_file = self.prior_params_file_older
                    pvprlog += [self.prior_params_file_older] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]
                elif pv in names_recent:
                    print('PRIOR STUFF: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_recent
                    pvprlog += [self.prior_params_file] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]
                else:
                    print('PRIOR STUFF: WARNING WARNING WARNING WARNING ' + pv + ' NOT FOUND')
                    print('PRIOR STUFF: MIGHT WANT TO CONSIDER DEFAULT VALUES')
                    pvprlog += ['PV not found in '+self.prior_params_file+' or '+self.prior_params_file_older] # for logging
                    pvprlog += [0]
                    

                # extract useful stats
                ave_m = filedata.get_value(pv, 'mean slope')
                ave_b = filedata.get_value(pv, 'mean intercept')
                width_m = filedata.get_value(pv, 'width slope')
                width_b = filedata.get_value(pv, 'width intercept')
                ave_res_m = filedata.get_value(pv, 'mean residual slope')
                ave_res_b = filedata.get_value(pv, 'mean residual intercept')
                width_res_m = filedata.get_value(pv, 'width residual slope')
                width_res_b = filedata.get_value(pv, 'width residual intercept')
                

                # build prior parameters from above stats

                # prior centroid
                prcentroid[i] = ave_m*self.energy + ave_b

                # prior widths
                prwidths[i] = (ave_res_m*self.energy + ave_res_b)
                
                # logging prior stuff
                pvprlog += [prcentroid[i],prwidths[i],ave_m,ave_b,width_m,width_b,ave_res_m,ave_res_b,width_res_m,width_res_b]
                self.model.prior_pv_info += [[pvprlog]]
                
                
            # end data reduction prior

        # override the prior centroid with starting position
        #prcentroid = self.start_dev_vals
        #print 'WARNING: overriding prior centroid with current value'
        #print '         CONSIDER USING PRIOR CENTROID'

        # finally, set the prior amplitude
        pramp = 0.1 # was 0.1
        
        # finally, compile all the prior params and LOAD THE MODEL
        print "Prior mean peak at point: ",prcentroid
        print "Prior mean widths: ",prwidths
        print "Prior mean amp: ",pramp
        prcentroid = np.array(prcentroid,ndmin=2)
        prwidths = np.array(prwidths, ndmin=2)
        self.model.prmean = multinormal_priorfcn # prior mean fcn
        self.model.prmeanp = [prcentroid, prwidths, pramp] # params of prmean fcn
        #print "self.model.prmean(prcentroid,self.model.prmeanp) = ",self.model.prmean(prcentroid,self.model.prmeanp)


    def terminate(self, devices):
        """
        Sets the position back to the location that seems best in hindsight.
        It's a good idea to run this at the end of the optimization, since
        Bayesian optimization tries to explore and might not always end in
        a good place.
        """
        print("TERMINATE", self.x_best)
        if(self.acq_func[0] == 'EI'):
            # set position back to something reasonable
            for i, dev in enumerate(devices):
                dev.set_value(self.x_best[i])
            #error_func(self.x_best)
        if(self.acq_func[0] == 'UCB'):
            # UCB doesn't keep track of x_best, so find it
            (x_best, y_best) = self.best_seen()
            for i, dev in enumerate(devices):
                dev.set_value(x_best[i])


    def minimize(self, error_func, x):
        # weighting for exploration vs exploitation in the GP at the end of scan, alpha array goes from 1 to zero
        #alpha = [1.0 for i in range(40)]+[np.sqrt(50-i)/3.0 for i in range(41,51)]
        inverse_sign = -1
        self.current_x = np.array(np.array(x).flatten(), ndmin=2)
        #self.current_y = [np.array([[inverse_sign*error_func(x)]])]
        self.X_obs = np.array(self.current_x)
        self.Y_obs = [np.array([[inverse_sign*error_func(x)]])]
        # iterate though the GP method
        #print("GP minimize",  error_func, x, error_func(x))
        for i in range(self.max_iter):
            # get next point to try using acquisition function
            x_next = self.acquire(self.alpha)
            #check for problems with the beam
            if self.check != None: self.check.errorCheck()

            y_new = error_func(x_next.flatten())
            #if self.kill:
            if self.opt_ctrl.kill:
                print 'Killing Bayesian optimizer...'
                #disable so user does not start another scan while the data is being saved
                break
            y_new = np.array([[inverse_sign *y_new]])

            #advance the optimizer to the next iteration
            #self.opt.OptIter(alpha=alpha[i])
            #self.OptIter() # no alpha

            # change position of interface and get resulting y-value

            x_new = deepcopy(x_next)
            #(x_new, y_new) = self.mi.getState()
            self.current_x = x_new
            #self.current_y = y_new

            # add new entry to observed data
            self.X_obs = np.concatenate((self.X_obs, x_new), axis=0)
            self.Y_obs.append(y_new)

            # update the model (may want to add noise if using testEI)
            self.model.update(x_new, y_new)# + .5*np.random.randn())


    def best_seen(self):
        """
        Checks the observed points to see which is predicted to be best.
        Probably safer than just returning the maximum observed, since the
        model has noise. It takes longer this way, though; you could
        instead take the model's prediction at the x-value that has
        done best if this needs to be faster.

        Not needed for UCB
        """
        if(self.acq_func[0] == 'UCB'):
            mu = self.Y_obs
        else:
            (mu, var) = self.model.predict(self.X_obs)

        (ind_best, mu_best) = max(enumerate(mu), key=op.itemgetter(1))
        return (self.X_obs[ind_best], mu_best)

    def acquire(self, alpha=None):
        """
        Computes the next point for the optimizer to try by maximizing
        the acquisition function. If movement per iteration is bounded,
        starts search at current position.
        """
        # look from best positions
        (x_best, y_best) = self.best_seen() # sort of a misnomer (see function best_seen)
        self.x_best = x_best
        x_curr = self.current_x[-1]
        #y_curr = self.current_y[-1]
        x_start = x_best
        
        # calculate length scales
        lengthscales = np.sqrt(0.5*np.exp(-self.model.covar_params[0][0])) # length scales from covar params
        ndim = x_curr.size # dimension of the feature space we're searching NEEDED FOR UCB
        try:
            nsteps = 1 + self.X_obs.shape[0] # acquisition number we're on  NEEDED FOR UCB
        except: 
            nsteps = 1

        #print "self.x_best = " + str(x_best)
        #print "self.current_x = " + str(self.current_x)
        #print "self.current_x[-1] = " + str(self.current_x[-1])
        
        # check to see if this is bounding step sizes
        if(self.iter_bound or True):
            if(self.bounds is None): # looks like a scale factor
                self.bounds = 1.0
            
            bound_lengths = 1. * lengthscales # 3x hyperparam lengths
            iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))
            
        else:
            iter_bounds = self.bounds

        #print "x_start = " + str(x_start)
        #print "BayesOpt.acquire - self.model.covar_params = " + str(self.model.covar_params)
        #print "self.model.covar_params[0] = " + str(self.model.covar_params[0])
        #print "iter_bounds = " + str(iter_bounds)

        # options for finding the peak of the acquisition function:
        optmethod = 'L-BFGS-B' # these 4 allow bounds
        #optmethod = 'BFGS'
        #optmethod = 'TNC'
        #optmethod = 'SLSQP'
        #optmethod = 'Powell' # these 2 don't
        #optmethod = 'COBYLA'
        maxiter = 1000 # max number of steps for one scipy.optimize.minimize call
        nproc = mp.cpu_count() # number of processes to launch minimizations on
        niter = 1 # max number of starting points for search
        niter_success = 1 # stop search if same minima for 10 steps
        tolerance = 1.e-4 # goal tolerance
        #nproc = 5*mp.cpu_count() # number of processes to launch minimization 
        
        # perturb start to break symmetry
        #x_start += np.random.randn(lengthscales.size)*lengthscales*1e-6

        # probability of improvement acquisition function
        if(self.acq_func[0] == 'PI'):
            print 'Using PI'
            aqfcn = negProbImprove
            fargs=(self.model, y_best, self.acq_func[1])

        # expected improvement acquisition function
        elif(self.acq_func[0] == 'EI'):
            print 'Using EI'
            aqfcn = negExpImprove
            fargs = (self.model, y_best, self.acq_func[1], alpha)
    
        # gaussian process upper confidence bound acquisition function
        elif(self.acq_func[0] == 'UCB'):
            print 'Using UCB'
            aqfcn = negUCB
            fargs = (self.model, ndim, nsteps, 0.01, 2.)

        # maybe something mitch was using once? (can probably remove)
        elif(self.acq_func[0] == 'testEI'):
            # collect all possible x values
            options = np.array(self.acq_func[2].iloc[:, :-1])
            (x_best, y_best) = self.best_seen()

            # find the option with best EI
            best_option_score = (-1,1e12)
            for i in range(options.shape[0]):
                result = negExpImprove(options[i],self.model,y_best,self.acq_func[1])
                if(result < best_option_score[1]):
                    best_option_score = (i, result)

            # return the index of the best option
            return best_option_score[0]

        else:
            print('Unknown acquisition function.')
            return 0
        
        try:
            # manual scan for diagnostics
            if False:
                nmax = 15.
                scale = 3.
                for i in scale*np.linspace(-1,1,nmax):
                    x = x_start + i * np.array(lengthscales,ndmin=2)
                    (y_mean, y_var) = self.model.predict(np.array(x, ndmin=2))
                    print i,x,y_mean,y_var,negExpImprove(x,self.model, y_best, self.acq_func[1], alpha)

            print 'iter_bounds = ',iter_bounds
            #print 'len(lengthscales) = ', len(lengthscales) 
            
            # plot heatmaps
            if True and len(lengthscales) == 2:
                
                print('Plotting heat maps.')
                
                #center_point = self.x_start # moving view
                center_point = self.start_dev_vals # static view
                rangex = center_point[0] + 5 * lengthscales[0] * np.array([-1,1]) #+ x_start[0]
                rangey = center_point[1] + 5 * lengthscales[1] * np.array([-1,1]) #+ x_start[1]
                
                try:
                    plotheatmap(self.model.predict,(),rangex,rangey,series=self.model.BV)
                except:
                    print 'Could not print prediction heatmap.'
                    pass

                try:
                    plotheatmap(aqfcn,fargs,rangex,rangey,series=self.model.BV)
                except:
                    print 'Could not print acquisition heatmap.'
                    pass
            
            if(self.multiprocessingQ):
                
                neval = 2*int(10.*2.**(ndim/12.))
                nkeep = 2*min(4,neval)

                print 'neval = ', neval,'\t nkeep = ',nkeep

                # parallelgridsearch generates pseudo-random grid, then performs a ICDF transform
                # to map to multinormal distrinbution centered on x_start and with widths given by hyper params
                v0s = parallelgridsearch(aqfcn,x_start,0.6*lengthscales,fargs,neval,nkeep)
                x0s = v0s[:,:-1] # for later testing if the minimize results are better than the best starting point
                v0best = v0s[0]
                #x0s = parallelgridsearch(aqfcn,x_start,lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))
                #x0s = np.vstack((x0s,parallelgridsearch(aqfcn,x_start,0.5*lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))))
                
                print 'self.model.covar_params = ',self.model.covar_params
                print 'self.model.noise_var = ',self.model.noise_var
                                
                x0s = np.vstack((x0s,np.array(x_curr))) # last point
                x0s = np.vstack((x0s,np.array(x_best))) # best so far
                print 'x0s = ', x0s

                if basinhoppingQ:
                    # use basinhopping
                    bkwargs = dict(niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':fargs,'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}}) # keyword args for basinhopping
                    res = parallelbasinhopping(aqfcn,x0s,bkwargs)

                else:
                    # use minimize
                    mkwargs = dict(bounds=iter_bounds, method=optmethod, options={'maxiter':maxiter}, tol=tolerance) # keyword args for scipy.optimize.minimize
                    res = parallelminimize(aqfcn,x0s,fargs,mkwargs,v0best)
                print 'res = ', res
                
            else: # single-processing
                if basinhoppingQ:
                    res = basinhopping(aqfcn, x_start,niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':(self.model, y_best, self.acq_func[1], alpha),'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}})
                
                else:
                    res = minimize(aqfcn, x_start, args=(self.model, y_best, self.acq_func[1], alpha), method=optmethod,tol=tolerance,bounds=iter_bounds,options={'maxiter':maxiter})

                res = res.x
                # end else
            #print 'res = ',res
        except:
            raise
        return np.array(res,ndmin=2) # return resulting x value as a (1 x dim) vector

# why is this class declared in BayesOptimization.py???
class HyperParams:
    def __init__(self, pvs, filename):
        self.pvs = pvs
        print 'HyperParams = ',self.pvs
        self.filename = filename
        pass

    def loadSeedData(self,filename, target):
        """ Load in the seed data from a matlab ocelot scan file.

        Input file should formated like OcelotInterface file format.
        ie. the datasets that are saved into the matlab data folder.

        Pulls out the vectors of data from the save file.
        Sorts them into the same order as this scanner objects pv list.
        The GP wont work if the data is in the wrong order and loaded data is not ordered.

        Args:
                filename (str): String for the .mat file directory.

        Returns:
                Matrix of ordered data for GP. [ len(num_iterations) x len(num_devices) ]
        """
        print
        dout = []
        if type(filename) == type(''):
            print "Loaded seed data from file:",filename
            #stupid messy formating to unest matlab format
            din = scipy.io.loadmat(str(filename))['data']
            names = np.array(din.dtype.names)
            for pv in self.pvs:
                pv = pv.replace(":","_")
                if pv in names:
                    x = din[pv].flatten()[0]
                    x = list(chain.from_iterable(x))
                    dout.append(x)

            #check if the right number of PV were pulled from the file
            if len(self.pvs) != len(dout):
                print "The seed data file device length unmatched with scan requested PVs!"
                print 'PV len         = ',len(self.pvs)
                print 'Seed data len = ',len(dout)
                self.parent.scanFinished()

            #add in the y values
            #ydata = din[self.objective_func_pv.replace(':','_')].flatten()[0]
            ydata = din[target.replace(':','_')].flatten()[0]
            dout.append(list(chain.from_iterable(ydata)))

        # If passing seed data from a seed scan
        else:
            print "Loaded Seed Data from Seed Scan:"
            din = filename
            for pv in self.pvs:
                if pv in din.keys():
                    dout.append(din[pv])
            dout.append(din[target])
            #dout.append(din[target])
                    #dout.append(din[target])

        #transpose to format for the GP
        dout = np.array(dout).T

        #dout = dout.loc[~np.isnan(dout).any(axis=1),:]
        dout = dout[~np.isnan(dout).any(axis=1)]

        #prints for debug
        print "[device_1, ..., device_N] detector"
        print self.pvs,target
        print dout
        print

        return dout

    def extract_hypdata(self, energy):
        key = str(energy)
        f = np.load(str(self.filename), fix_imports=True, encoding='latin1')
        filedata = f[0][key]
        return filedata

    #def loadHyperParams(self, filename, energy, detector, pvs, multiplier = 1):
    def loadHyperParams(self, filename, energy, detector, pvs, vals, multiplier = 1):
        """
        Method to load in the hyperparameters from a .npy file.
        Sorts data, ordering parameters with this objects pv list.
        Formats data into tuple format that the GP model object can accept.
        ( [device_1, ..., device_N ], coefficent, noise)
        Args:
                filename (str): String for the file directory.
                energy:
        Returns:
                List of hyperparameters, ordered using the UI's "self.pvs" list.
        """
        #Load in a npy file containing hyperparameters binned for every 1 GeV of beam energy
        extention = self.filename[-4:]
        print ('hyper parameter filename = ', filename)
        
        # use prior mean scrapes to choose length scales?
        if extention == ".pkl":
            filedata = pd.read_pickle(filename)

            names = filedata.T.keys()
            pvs = [pv.replace(":","_") for pv in pvs]
            energy = float(energy)
            hyps = []
            match_count=0
            for i, pv in enumerate(pvs):

                if pv in names and filedata.get_value(pv, 'number of points fitted')>10:
                    print('length scale stuff')
                    print (pv + " AUGUST DATA LOOKS GOOD")

                    ave_m = filedata.get_value(pv, 'mean slope')
                    ave_b = filedata.get_value(pv, 'mean intercept')
                    std_m = filedata.get_value(pv, 'width slope')
                    std_b = filedata.get_value(pv, 'width intercept')
                    ave = ave_m*energy + ave_b
                    std = std_m*energy + std_b
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper params:", pv, ave, std, hyp)
                    match_count+=1
                elif pv in names and filedata.get_value(pv, 'number of points fitted')<=10:
                    PRINT('LENGTH SCALE STUFF:')
                    print(pv + ' AUGUST DATA TOO SPARSE, USING ALL 2017')
                    ave_m = filedata2.get_value(pv, 'mean slope')
                    ave_b = filedata2.get_value(pv, 'mean intercept')
                    std_m = filedata2.get_value(pv, 'width slope')
                    std_b = filedata2.get_value(pv, 'width intercept')
                    ave = ave_m*energy + ave_b
                    std = std_m*energy + std_b
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper params:", pv, ave, std, hyp)
                    match_count+=1
                else: # default values
                    print('LENGTH SCALE STUFF:')
                    print('USING DEFAULT VALUES')
                    ave = float(vals[i])
                    std = np.sqrt(abs(ave))
                    print(ave,std)
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper params:", pv, ave, std, hyp)
                    print ("from values: ", float(vals[i]))
                    match_count+=1 #Should this be incremented even though it is in the 'else' condition??
                    
        # default Ocelot hyper parameters
        else:
            if extention == ".npy":
                #get current L3 beam
                if len(energy) is 3: key = energy[0:1]
                if len(energy) is 4: key = energy[0:2]
                print "Loading raw data for",key,"GeV from",filename
                print energy, pvs
                f = np.load(str(filename))
                filedata = f[0][key]

            #sort list to match the UIs PV list order
            #if they are loaded in the wrong order, the optimzer will get the wrong params for a device
            keys = []
            hyps = []
            match_count = 0
            #for pv in pvs:
            print "GP/BayesOptimization.py:HyperParams.loadHyperParams - pvs = " + str(pvs)
            for i, pv in enumerate(pvs):
                names = filedata.keys()
                if pv in names:
                    keys.append(pv)
                    ave = float(filedata[pv][0])
                    std = float(filedata[pv][1])
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper params:", pv, ave, std, hyp)
                    match_count+=1
                else: # default values
                    ave = float(vals[i])
                    std = np.sqrt(abs(ave))
                    print(ave,std)
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper params:", pv, ave, std, hyp)
                    print ("from values: ", float(vals[i]))
                    match_count+=1 #Should this be incremented if pv is not in names??

        if match_count != len(self.pvs):
            # TODO: what is it?
            # self.parent.scanFinished()
            raise Exception("Missing PV(s) in hyperparameter file " + filename)

        # WARNING
        # get the current mean and std of the chosen detector, definitely needs to change
        obj_func = detector.get_value()[0]
        print('obj_func = ',obj_func)
        try:
            #std = np.std(  obj_func[(2799-5*120):-1])
            #ave = np.mean( obj_func[(2799-5*120):-1])
            std = np.std(  obj_func[-120:])
            ave = np.mean( obj_func[-120:])
        except:
            print "Detector is not a waveform, Using scalar for hyperparameter calc"
            print "Also check GP/BayesOptimization.py:HyperParams.loadHyperParams near line 722"
            ave = obj_func
            # Hard code in the std when obj func is a scalar
            # Not a great way to deal with this, should probably be fixed
            std = 0.1
            
        print('WARNING: overriding amplitude hyper param')
        ave = 1.

        print ("DETECTOR AVE", ave)
        print ("DETECTOR STD", std)

        coeff = self.calcAmpCoeffHP(ave, std)
        noise = self.calcNoiseHP(ave, std)

        dout = ( np.array([hyps]), coeff, noise )
        #prints for debug
        print()
        print ("Calculated Hyperparameters ( [device_1, ..., device_N ], amplitude coefficent, noise coefficent)")
        print()
        for i in range(len(hyps)):
            print(self.pvs[i], hyps[i])
        print ("AMP COEFF   = ", coeff)
        print ("NOISE COEFF = ", noise)
        print()
        return dout

    def calcLengthScaleHP(self, ave, std, c = 1.0, multiplier = 1, pv = None):
        """
        Method to calculate the GP length scale hyperparameters using history data
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        Args:
                ave (float): Mean of the device, binned around current machine energy
                std (float): Standard deviation of the device
                c   (float): Scaling factor to change the output to be larger or smaller, determined empirically
                pv  (str): PV input string to scale hyps depending on pv, not currently used
        Returns:
                Float of the calculated length scale hyperparameter
        """
        #for future use
        if pv is not None:
            #[pv,val]
            pass
        #+- 1 std around the mean
        #hi  = ave+std
        #lo  = ave-std
        #hyp = -2*np.log( ( ( multiplier*c*(hi-lo) ) / 4.0 ) + 0.01 )
        hyp = -2*np.log( ( ( multiplier*c*std ) / 2.0 ) + 0.01 )
        return hyp

    def calcAmpCoeffHP(self, ave, std, c = 0.5):
        """
        Method to calculate the GP amplitude hyperparameter
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        First we tried using the standard deviation to calc this but we found it needed to scale with mean instead
        Args:
                ave (float): Mean of of the objective function (GDET or something else)
                std (float): Standard deviation of the objective function
                c (float): Scaling factor to change the output to be larger or smaller, determined empirically
        Returns:
                Float of the calculated amplitude hyperparameter
        """
        #We would c = 0.5 to work well, could get changed at some point
        hyp2 = np.log( ( ((c*ave)**2) + 0.1 ) )
        #hyp2 = np.log( ave + 0.1 )
        return hyp2

    def calcNoiseHP(self, ave, std, c = 1.0):
        """
        Method to calculate the GP noise hyperparameter
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        Args:
                ave (float): Mean of of the objective function (GDET or something else)
                std (float): Standard deviation of the objective function
                c (float): Scaling factor to change the output to be larger or smaller, determined empirically
        Returns:
                Float of the calculated noise hyperparameter
        """
        hyp = np.log((c*std / 4.0) + 0.01)
        #hyp = np.log(std + 0.01)
        return hyp


def negProbImprove(x_new, model, y_best, xi):
    """
    The probability of improvement acquisition function. Initial testing
    shows that it performs worse than expected improvement acquisition
    function for 2D scans (at least when alpha==1 in the fcn below). Alse
    performs worse than EI according to the literature.
    """
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))
    diff = y_mean - y_best - xi
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    return -norm.cdf(Z)

def negExpImprove(x_new, model, y_best, xi, alpha=1.0):
    """
    The common acquisition function, expected improvement. Returns the
    negative for the minimizer (so that EI is maximized). Alpha attempts
    to control the ratio of exploration to exploitation, but seems to not
    work well in practice. The terminate() method is a better choice.
    """
    (y_mean, y_var) = model.predict(np.array(x_new, ndmin=2))
    diff = y_mean - y_best - xi
    
    # Nonvectorizable. Can prob use slicing to do the same.
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    EI = diff * norm.cdf(Z) + np.sqrt(y_var) * norm.pdf(Z)
    #print(x_new, EI)
    return alpha * (-EI) + (1. - alpha) * (-y_mean)

# old version
#def negUCB(x_new, model, mult):
    #"""
    #The upper confidence bound acquisition function. Currently only partially
    #implemented. The mult parameter specifies how wide the confidence bound
    #should be, and there currently is no way to compute this parameter. This
    #acquisition function shouldn't be used until there is a proper mult.
    #"""
    #(y_new, var) = model.predict(np.array(x_new,ndmin=2))

    #UCB = y_new + mult * np.sqrt(var)
    #return -UCB

# GP upper confidence bound
# original paper: https://arxiv.org/pdf/0912.3995.pdf
# tutorial: http://www.cs.ubc.ca/~nando/540-2013/lectures/l7.pdf
def negUCB(x_new, model, ndim, nsteps, nu = 1., delta = 1.):
    """
    GPUCB: Gaussian process upper confidence bound aquisition function
    Default nu and delta hyperparameters theoretically yield "least regret".
    Works better than "expected improvement" (for alpha==1 above) in 2D.

    input params
    x_new: new point in the dim-dimensional space the GP is fitting
    model: OnlineGP object
    ndim: feature space dimensionality (how many devices are varied)
    nsteps: current step number counting from 1
    nu: nu in the tutorial (see above)
    delta: delta in the tutorial (see above)
    """

    #ndim = model.nin # problem space dimensionality
    #nsteps = model.nupdates + 1 # current step number
    if nsteps==0: nsteps += 1
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))

    tau = 2.*np.log(nsteps**(0.5*ndim+2.)*(np.pi**2.)/3./delta)
    GPUCB = y_mean + np.sqrt(nu * tau * y_var)

    return -GPUCB

# Thompson sampling
