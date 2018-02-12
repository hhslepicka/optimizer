# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import multiprocessing as mp
#import math
from scipy.special import erfinv
#from hammersley import hammersley
from chaospy_sequences import create_hammersley_samples
                    
# see here https://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
# and here https://stackoverflow.com/questions/37060091/multiprocessing-inside-function

def mworker(f,x0,fargs,margs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
#    f,x0,fargs,margs = args
    #print 'worker: fargs = ',fargs
    #print 'worker: margs = ',margs
    res = minimize(f, x0, args = fargs, **margs)
    #return [res.x, res.fun]
    out_q.put([[res.x, res.fun[0][0]]])

# parallelize minimizations using different starting positions using multiprocessing, scipy.optimize.minimize
def parallelminimize(f,x0s,fargs,margs):
    # f is fcn to minimize
    # x0s are positions to start search from
    # fargs are arguments to pass to f
    # margs are arguments to pass to scipy.optimize.minimize
    
    # Each process will get a queue to put its result in
    out_q = mp.Queue()
        
    # arguments to loop over
    args = [(f,x0,fargs,margs,out_q) for x0 in x0s]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    # also could try concurrent futures
#    import multiprocessing
#    pool = multiprocessing.Pool()
#    res = np.array(pool.map(minimizeone, args))
#    res = np.array(pool.map(l, range(10)))
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
    
    nprocs = len(x0s)
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=mworker,
                args=args[i])
        procs.append(p)
        p.start()

    res = [];
    for i in range(nprocs):
        res += out_q.get()

    for p in procs:
        p.join()
        
    res = np.array(res)
    print 'res = ', res
    res = res[res[:,1]==np.min(res[:,1])][0]
    print 'selected min is ',res[1]
    res = np.array(res[0])

    return res

def bworker(f,x0,bkwargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    res = basinhopping(f, x0, **bkwargs)
    out_q.put([[res.x, res.fun[0][0]]])

# parallelize minimizations using different starting positions using multiprocessing, scipy.optimize.minimize
def parallelbasinhopping(f,x0s,bkwargs):
    # f is fcn to minimize
    # x0s are positions to start search from
    # fargs are arguments to pass to f
    # margs are arguments to pass to scipy.optimize.minimize
    
    # Each process will get a queue to put its result in
    out_q = mp.Queue()
        
    # arguments to loop over
    args = [(f,x0,bkwargs,out_q) for x0 in x0s]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    # also could try concurrent futures
#    import multiprocessing
#    pool = multiprocessing.Pool()
#    res = np.array(pool.map(minimizeone, args))
#    res = np.array(pool.map(l, range(10)))
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
    
    nprocs = len(x0s)
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=bworker,
                args=args[i])
        procs.append(p)
        p.start()

    res = [];
    for i in range(nprocs):
        res += out_q.get()

    for p in procs:
        p.join()
        
    res = np.array(res)
    #print 'res = ', res
    res = res[res[:,1]==np.min(res[:,1])][0]
    #print 'selected min is ',res
    res = np.array(res[0])

    return res


def mapworker(f,x,fargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    #print 'f = ',f,'\tx = ',x,'\tfargs = ',fargs,'\tf(x, *fargs) = ',f(x, *fargs)
    #out_q.put([[x,f(x, *fargs)]])
    out_q.put([[f(x, *fargs)]])

# yuno stock have python?!
def parallelmap(f,xs,fargs):
    # f is fcn to map to
    # xs is list of coords to eval
    # fargs are arguments to pass to f
    
    # Each process will get a queue to put its result in
    out_q = mp.Queue()
        
    # arguments to loop over
    args = [(f,x,fargs,out_q) for x in xs]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
    
    nprocs = len(xs)
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=mapworker,
                args=args[i])
        procs.append(p)
        p.start()

    res = [];
    for i in range(nprocs):
        res += out_q.get()

    for p in procs:
        p.join()
        
    # sort by argument order passed (seems presorted but just in case)
    #res = res[res[:,0].argsort()] # this doesn't work in genreal; must mix back in xs

    return res 

    
    
def eworker(f,x,fargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    res = f(x, *fargs)
    out_q.put(np.hstack((x, res[0][0])))

# eval function over a range of initial points neval and return the nkeep lowest function evals
def parallelgridsearch(f,x0,lengths,fargs,neval,nkeep):
    # f is fcn to minimize
    # x0 is center of the search
    # lengths is an array of length scales
    # fargs are arguments to pass to f
    # neval is the number of points to evaluate the function on
    # nkeep is the number of the neval points to keep
    
    if nkeep > neval: nkeep = neval
    
    # Each process will get a queue to put its result in
    out_q = mp.Queue()
    
    # generate points to search
    ndim = len(lengths)
    nevalpp = neval + 1
    #x0s = np.array([hammersley(i,ndim,nevalpp) for i in range(1,nevalpp)]) # hammersley uniform in all dims
    #x0s = np.vstack(parallelmap(hammersley, range(1,nevalpp), (ndim,nevalpp)))
    x0s = create_hammersley_samples(order=neval, dim=ndim).T
    x0s = np.sqrt(2)*erfinv(-1+2*x0s) # normal in all dimensions
    x0s = np.transpose(np.array(lengths,ndmin=2).T * x0s.T) # scale each dimension by it's lenghth scale
        
    # arguments to loop over
    args = [(f,x0,fargs,out_q) for x0 in x0s]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    # also could try concurrent futures
#    import multiprocessing
#    pool = multiprocessing.Pool()
#    res = np.array(pool.map(minimizeone, args))
#    res = np.array(pool.map(l, range(10)))
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
    
    nprocs = neval
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=eworker,
                args=args[i])
        procs.append(p)
        p.start()

    res = np.array(out_q.get());
    for i in range(nprocs-1):
        res = np.vstack((res,out_q.get()))

    for p in procs:
        p.join()
        
    res = np.array(res)
    res = np.sort(res)
    res = res[res[:,-1]<=res[nkeep-1,-1]] # list of nkeep coords and function evals there
    
    print 'nkeep coords and function evals: ', res

    #return res # return coords and fcn evals
    return res[:,:-1] # return just coords
