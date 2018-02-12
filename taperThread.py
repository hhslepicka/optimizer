#thread spawned by OcelotInterface.py when a scan is launched with a taper parameter selected
#polls taper PVs for change, updates undulator line to reflect new parameters, blocks main optimization loop while segments are moving.

import threading
import numpy as np
import epics
#import epicsGet
import matplotlib.pyplot as plt

from PyQt4.QtCore import QTimer
from scipy import optimize as op
from time import sleep

#watches taper PVs and sends setTaper commands when they change
class taperWatcher(threading.Thread):
    def __init__(

            self,
            updateTime=.1, #time between calls to checkTaperPVs
            ):

        super(taperWatcher, self).__init__()
        self.updateTime = updateTime
        self.readyPV = 'SIOC:SYS0:ML00:CALCOUT805'      #pv = 0 when segments are moving, 1 when they are not moving
        self.getter = epicsGet.epicsGet()
        self.kill = False

    def run(self):

        #main
        self.taper = Taper(self)
        self.taperPVs = self.taper.getPVs()
        self.taper.setTaper(self.taper.p)       #sets undulator line to initial fit of starting taper
        self.oldTaper = self.taper.p            #saves old/initial taper parameters
        print 'Starting taper thread'
        while not self.kill:
            self.checkTaperPVs()
            sleep(self.updateTime)

    #poll taperPVs and send setTaper command when they change
    def checkTaperPVs(self):
        newTaper = []
        undsOut = []
        tries = 5                                       #number of times to try to set segments that do not move into place after trim
        for pv in self.taperPVs:                        #loops through pvs and make list of current parameters
            newTaper.append(self.getter.caget(pv))
        if(not newTaper == self.oldTaper):              #if parameters/pvs have changed since last time polled
        #        print newTaper, '!=', self.oldTaper
            while(tries > 0):
                undsOut = self.taper.setTaper(newTaper, undsOut)
                if(len(undsOut) > 0):
                    print "some segments still out, trying again; ", tries, " tries left"
                    tries = tries - 1;
                else: break
            self.oldTaper = newTaper
        epics.caput(self.readyPV, 1)

#initialize pvs used by program; set taper pvs to current taper parameters if initPV True
class Taper():
    def __init__(self, parent = None, initPVs = False):
        self.getter = epicsGet.epicsGet()
        self.actpv = []
        self.despv = []
        self.trimpv = []
        self.initialK = []
        self.movePVs = []
        self.numK = 32
        self.undsNotUsedIndex = [0, 8,15]
        self.undsNotUsedIndex = self.undsNotUsedIndex + self.undsOut()
        self.defaultStartParam = [-0.0003, 10, -0.0001] #used by curve fit in getTaperCoefficients
        self.pvs = ['PHYS:ACR0:OCLT:LINAMP', 'PHYS:ACR0:OCLT:POSTSATSTART', 'PHYS:ACR0:OCLT:POSTSATAMP']
        self.outOfTolPV = 'SIOC:SYS0:ML00:CALCOUT806'
        self.parent = parent
        self.debug = True   #set to true if you don't want und segments to actually move
        for i in range(1, self.numK+1):
            if(i-1 not in self.undsNotUsedIndex):
                self.actpv.append('USEG:UND1:' + str(i) + '50:KACT')
                self.despv.append('USEG:UND1:' + str(i) + '50:KDES')
                self.trimpv.append('USEG:UND1:' + str(i) + '50:TRIM.PROC')
                self.movePVs.append('USEG:UND1:' + str(i) + '50:TM1MOTOR.DMOV')
                self.movePVs.append('USEG:UND1:' + str(i) + '50:TM2MOTOR.DMOV')
                self.initialK.append(self.getter.caget(self.despv[-1]))
        self.firstK = self.initialK[0]
        self.p = self.getTaperCoefficients(self.initialK) # [linamp postsatstart postsatampl]
        if(initPVs):
            for (pv, param) in zip(self.pvs, self.p):
                if pv == 'PHYS:ACR0:OCLT:POSTSATSTART':
                    param = int(round(param))
                epics.caput(pv, param)

    def get_value(self):
        return self.p


    #fit line to parameters p and set undulator line; block while segments are moving; if moveUnds is not empty, move all segments not in undsNotUsedIndex; otherwise, move only moveUnds
    def setTaper(self, p, moveUnds = []):
        x = np.array(range(0, self.numK))
        x = np.delete(x, self.undsNotUsedIndex)
        newTaper = self.taper(x, *p)
        if(not self.withinTol(newTaper)):
            epics.caput(self.outOfTolPV, 1)
            return
        epics.caput(self.outOfTolPV, 0)
        print 'Setting Taper: ', newTaper
        if(len(moveUnds) == 0 ): move = range(0, len(newTaper))
        else:
            move = moveUnds
        for i in move:
            if(self.debug):
                print 'caput ', self.despv[i], ' ' , newTaper[i]
                print 'caput ', self.trimpv[i], ' ', 1
                continue
            epics.caput(self.despv[i], newTaper[i])
            epics.caput(self.trimpv[i], str(1))
        while(self.stillMoving()):
            sleep(.1)
            if(self.parent.kill): break
        #check if any valid segments didn't move into place
        stillOut = self.undsOut()
        fundsOut = []
        for i in range(0, len(newTaper)):
            if(i in stillOut and i not in self.undsNotUsedIndex):
                stillOut.append(i)
        return stillOut


    def taper(self,x, linAmp, postSatStart, postSatAmp):    #calc individual k values from given set of parameters
        def heaviside(x):
            return 0.5*(np.sign(x) + 1)
        #print self.firstK + linAmp*x + heaviside(x -postSatStart)*postSatAmp*(x - postSatStart)**2
        return self.firstK - abs(linAmp)*x - heaviside(x -postSatStart)*abs(postSatAmp)*(x - postSatStart)**2

    #get taper parameters by fitting k values passed to it.
    def getTaperCoefficients(self, k):
        firstK = self.firstK
        x = np.array(range(0, self.numK))
        x = np.delete(x, self.undsNotUsedIndex)
        p = op.curve_fit(self.taper, x, k, self.defaultStartParam)
        return p[0].tolist()

    def getPVs(self):
        return self.pvs;
    #check if undulator segments are moving
    def stillMoving(self):
        sleep(.1)
        for pv in self.movePVs:
            if(not(self.getter.caget(pv) == 1)):
                print pv, ' moving'
                return True;
        return False

    #check if undulators that should be in (i.e. not in undsNotUsedIndex) are out, returns those segments
    def undsOut(self):
        outPVs = []
        for i in range(1, self.numK+1):
            if(i-1 not in self.undsNotUsedIndex):
                statPV = 'USEG:UND1:' + str(i) + '50:LOCATIONSTAT'
                if(not(self.getter.caget(statPV) == 1)):
                    outPVs.append(i-1);
        return outPVs

    #check if k values are within tolerance of undulator
    def withinTol(self, newTaper):
        for pv, newK in zip(self.despv, newTaper):
            hilim = self.getter.caget(pv + '.HOPR')
            lolim = self.getter.caget(pv + '.LOPR')
            if(newK > hilim or newK < lolim):
                return False

        return True

    #reset und segments to initial taper from when Taper object was initialized
    def resetTaper(self):
        print('Resetting Taper')
        for i in range(0, len(self.initialK)):
            if(self.debug):
                print 'caput ', self.despv[i], ' ' , self.initialK[i]
                print 'caput ', self.trimpv[i], ' ', 1
                continue

            epics.caput(self.despv[i], self.initialK[i])
            epics.caput(self.trimpv[i], str(1))
        self.p = self.getTaperCoefficients(self.initialK) # [linamp postsatstart postsatampl]
        for (pv, param) in zip(self.pvs, self.p):
            epics.caput(pv, param)

    #hardcoded means/std used by norm functions for strength scaling
    def taperParams(self):
        taperParams = {}
        taperParams[self.pvs[0]] = [-0.000340994106915,0.000010269781282] #linamp
        taperParams[self.pvs[1]] = [10.954197518715358,.40929950080439]   #postsatstart
        taperParams[self.pvs[2]] = [-0.000094961380596,0.000003714580572]  #postsatampl
        ## same guys for different energies
        # 14 GeV
        #taperParams[self.pvs[0]] = [-0.00016168306689225289, 2.2758259890182469e-05] #linamp
        #taperParams[self.pvs[1]] = [15.439435422556567, 0.072016973387182048]   #postsatstart
        #taperParams[self.pvs[2]] = [-0.0002282192378443941, 9.4762326385699058e-06]  #postsatampl
        #
        # 3 GeV
        #taperParams[self.pvs[0]] = [-0.00022849389384432237, 9.436415472175746e-06] #linamp
        #taperParams[self.pvs[1]] = [9.0593460328343767, 0.30601754025349792]   #postsatstart
        #taperParams[self.pvs[2]] = [-0.00015737605882482986, 3.8394413633286863e-06]  #postsatampl
        return taperParams

    #check if pv passed is a taperpv
    def isTaperPV(self, pv):
        return pv in self.pvs
