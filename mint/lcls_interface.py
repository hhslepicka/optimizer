# -*- coding: utf-8 -*-
"""
Machine interface file for the LCLS to ocelot optimizer

Tyler Cope, 2016
"""
from PyQt5.QtGui import QPixmap
import numpy as np
try:
    import epics
except:
    print('No Module named epics. LCLSMachineInterface will not work. Try simulation mode instead')
import time
import sys
import math
import os
#logbook imports
from re import sub
from xml.etree import ElementTree
from shutil import copy
from datetime import datetime
from os import system
try:
    import Image
except:
    try: 
        from Pillow import Image
    except:
        try:
            from PIL import Image
        except:
            print('No Module named Image')
import sqlite3
from sqlite3 import Error

class LCLSMachineInterface():
    """ Start machine interface class """

    def __init__(self):
        # interface name
        self.name = 'LCLSMachineInterface'
        
        # default statistic to tune on
        self.stat_name = 'Mean'
        
        """ Initialize parameters for the scanner class. """
        self.initErrorCheck() #Checks for errors and trimming

    #=================================================================#
    # -------------- Original interface file functions -------------- #
    #=================================================================#

    def initErrorCheck(self):
        """
        Initialize PVs and setting used in the errorCheck method.
        """
        #setup pvs to check
        self.error_bcs      = "BCS:MCC0:1:BEAMPMSV"
        self.error_mps      = "SIOC:SYS0:ML00:CALCOUT989"
        self.error_guardian = "SIOC:SYS0:ML00:AO466"
        self.error_und_tmit = "BPMS:UND1:3290:TMITTH"

        #pv to bypass the error pause
        self.error_bypass  = "SIOC:SYS0:ML00:CALCOUT990"
        self.error_tripped = "SIOC:SYS0:ML00:CALCOUT991"

        #set the unlatch pv to zero
        epics.caput(self.error_bypass, 0)
        epics.caput(self.error_tripped,0)

    def errorCheck(self):
        """
        Method that check the state of BCS, MPS, Gaurdian, UND-TMIT and pauses GP if there is a problem.
        """
        while 1:
            #check for bad state
            if epics.caget(self.error_bypass)     == 1:
                out_msg="Bypass flag is TRUE"
            elif epics.caget(self.error_bcs)      != 1:
                out_msg="BCS tripped"
            elif epics.caget(self.error_mps)      != 0:
                out_msg="MPS tripped"
            elif epics.caget(self.error_guardian) != 0:
                out_msg="Gaurdian tripped"
            elif epics.caget(self.error_und_tmit) < 5.0e7:
                out_msg="UND Tmit Low"
            else:
                out_msg='Everything Okay'

            #exit if the stop button is set
            #if not self.mi.getter.caget("SIOC:SYS0:ML03:AO702"):
            if not epics.caget("SIOC:SYS0:ML03:AO702"):
                break

            #set the error check message
            epics.caput ("SIOC:SYS0:ML00:CA000",out_msg)
            print out_msg

            #break out if error check is bypassed
            if (out_msg=="Bypass flag is TRUE"):
                break

            #break out if everything is okay
            if (out_msg=="Everything Okay"):
                epics.caput(self.error_tripped,0)
                break
                #return
            else:
                epics.caput(self.error_tripped,1)
            time.sleep(0.1)

    def get_sase(self, datain, points=None):
        """
        Returns data for the ojective function (sase) from the selected detector PV.

        At lcls the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.

        Args:
                seconds (float): Variable input on how many seconds to average data

        Returns:
                Float of SASE or other detecor measurment
        """

        ## more sensitive statistic:
        #try:
            #dataout = np.percentile(datain[-(points):],90) # 90th percentile
            ##dataout   = np.std( datain[-(points):]) # tune on standard deviation
            #sigma   = np.std( datain[-(points):])
        #except: #if average fails use the scalar input
            #print "Detector is not a waveform PV, using scalar value"
            #dataout = datain
            #sigma   = -1
        
        # standard run:
        try:
            if self.stat_name == 'Median':
                statistic = np.median(datain[-int(points):])
            elif self.stat_name == 'Standard deviation':
                statistic = np.std(datain[-int(points):])
            elif self.stat_name == 'Median deviation':
                median = np.median(datain[-int(points):])
                statistic = np.median(np.abs(datain[-int(points):]-median))
            elif self.stat_name == 'Max':
                statistic = np.max(datain[-int(points):])
            elif self.stat_name == 'Min':
                statistic = np.min(datain[-int(points):])
            elif self.stat_name == '80th percentile':
                statistic = np.percentile(datain[-int(points):],80)
            elif self.stat_name == 'average of points > mean':
                dat_last = datain[-int(points):]
                percentile = np.percentile(datain[-int(points):],50)
                statistic = np.mean(dat_last[dat_last>percentile])
            elif self.stat_name == '20th percentile':
                statistic = np.percentile(datain[-int(points):],20)
            else:
                self.stat_name = 'Mean'
                statistic = np.mean(datain[-int(points):])
            # check if this is even used:
            sigma   = np.std( datain[-int(points):])
        except: #if average fails use the scalar input
            print "Detector is not a waveform PV, using scalar value"
            statistic = datain
            sigma   = -1
            
        print self.stat_name, ' of ', datain[-int(points):].size, ' points is ', statistic, ' and standard deviation is ', sigma

        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #print "WARNING returning negative of objective WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING "
        #statistic = -statistic
            
        return statistic, sigma

    def get_charge_current(self):
        charge = self.get_value('SIOC:SYS0:ML00:CALC252')
        current = self.get_value('BLEN:LI24:886:BIMAX')
        return charge, current

    def dataDelay(self, obj_func, points=None):
        datain = self.get_value(obj_func)
        if hasattr(datain, '__len__'): #Find out if it's an array or scalar
            rate = self.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
            try:
                collectionTime = points/rate
            except:
                print('unable to calculate collection time based on points requested or rate')
                collectionTime = 1
            print('collecting data for ', collectionTime, ' seconds')
        else:
            collectionTime = 0
        return collectionTime

    def get(self, pv):
        return epics.PV(str(pv), connection_timeout = 0.1).get()

    def put(self, pv, val):
        epics.caput(str(pv), val)

    def get_value(self, device_name):
        """
        Getter function for lcls.

        Args:
                device_name (str): String of the pv name used in caget

        Returns:
                Data from caget, variable data type depending on PV
        """
        ct = 0
        while 1:
            try:
                return epics.caget(str(device_name))
            except:
                print("Error retriving ca data! Tyring to caget data again")
                time.sleep(.05)
            ct+=1
        if ct > 3:
            raise Exception("Too many caget trys ,exiting")
            return None


    def set_value(self, device_name, val):
        """
        Setter function for lcls.

        Args:
                device_name (str): String of the pv name used in caput
                val (variable): Value to caput to device, variable data type depending on PV
        """
        epics.caput(device_name, val)

    def get_energy(self):
        return epics.caget("BEND:DMP1:400:BDES")

    def setListener(self, state):
        """
        Method to set epics flag inducating that this GUI is running.

        Args:
                state (bool): Bools to set the PV stats flag true or false
        """
        #watcher cud flag
        try:
            epics.caput("PHYS:ACR0:OCLT:OPTISRUNNING", state)
        except:
            print "No watcher cud PV found!"
        #listener application flag
        try:
            epics.caput("SIOC:SYS0:ML03:AO702" ,state)
        except:
            print "No listener PV found!"

        #sets the hostname env to another watcher cud PV
        try:
            opi = os.environ['HOSTNAME']
            epics.caput("SIOC:SYS0:ML00:CA999",opi)
        except:
            print "No OPI enviroment variable found"

    #=======================================================#
    # -------------- Normalization functions -------------- #
    #=======================================================#

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
        val = self.get_value(device)
        tol = (val*percent)
        lim_lo = val-tol
        lim_hi = val+tol
        limits = [lim_lo,lim_hi]
        return limits

    #=======================================================#
    # ------------------- Log Booking --------------------- #
    #=======================================================#

    def logbook(self,objective_func_pv, objective_func, winID, extra_log_text='default'):
        """
        Send a screenshot to the physics logbook.

        Args:
                extra_log_text (str): string to set if verbose text should be printed to logbook. 'default' prints only gain and algorithm
        """
        #Put an extra string into the logbook function

        log_text = "Gain ("+str(objective_func_pv)+"): "+str(round(objective_func.values[0],4))+" > "+str(round(objective_func.values[-1],4))
        if extra_log_text != 'default':
            log_text = log_text+'\n'+str(extra_log_text)
        curr_time = datetime.now()
        timeString = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
        log_entry = ElementTree.Element(None)
        severity  = ElementTree.SubElement(log_entry, 'severity')
        location  = ElementTree.SubElement(log_entry, 'location')
        keywords  = ElementTree.SubElement(log_entry, 'keywords')
        time      = ElementTree.SubElement(log_entry, 'time')
        isodate   = ElementTree.SubElement(log_entry, 'isodate')
        log_user  = ElementTree.SubElement(log_entry, 'author')
        category  = ElementTree.SubElement(log_entry, 'category')
        title     = ElementTree.SubElement(log_entry, 'title')
        metainfo  = ElementTree.SubElement(log_entry, 'metainfo')
        imageFile = ElementTree.SubElement(log_entry, 'link')
        imageFile.text = timeString + '-00.ps'
        thumbnail = ElementTree.SubElement(log_entry, 'file')
        thumbnail.text = timeString + "-00.png"
        text      = ElementTree.SubElement(log_entry, 'text')
        log_entry.attrib['type'] = "LOGENTRY"
        category.text = "USERLOG"
        location.text = "not set"
        severity.text = "NONE"
        keywords.text = "none"
        time.text = curr_time.strftime("%H:%M:%S")
        isodate.text =  curr_time.strftime("%Y-%m-%d")
        metainfo.text = timeString + "-00.xml"
        fileName = "/tmp/" + metainfo.text
        fileName=fileName.rstrip(".xml")
        log_user.text = " "
        title.text = unicode("Ocelot Interface")
        text.text = log_text
        if text.text == "": text.text = " " # If field is truly empty, ElementTree leaves off tag entirely which causes logbook parser to fail
        xmlFile = open(fileName+'.xml',"w")
        rawString = ElementTree.tostring(log_entry, 'utf-8')
        parsedString = sub(r'(?=<[^/].*>)','\n',rawString)
        xmlString=parsedString[1:]
        xmlFile.write(xmlString)
        xmlFile.write("\n")  # Close with newline so cron job parses correctly
        xmlFile.close()
        self.screenShot(fileName,'png', winID)
        path = "/u1/lcls/physics/logbook/data/"
        copy(fileName+'.ps', path)
        copy(fileName+'.png', path)
        copy(fileName+'.xml', path)

    def screenShot(self,filename,filetype, winID):
        """
        Takes a screenshot of the whole gui window, saves png and ps images to file

        Args:
                fileName (str): Directory string of where to save the file
                filetype (str): String of the filetype to save
        """
        s = str(filename)+"."+str(filetype)
        p = QPixmap.grabWindow(winID)
        p.save(s, 'png')
        im = Image.open(s)
        im.save(s[:-4]+".ps")
        p = p.scaled(465,400)
        p.save(str(s), 'png')


    def logTextVerbose(self, objective_func_pv, objective_func, trim_delay, numPulse, norm_amp_coeff, SeedScanBool, name_opt, winID):
        """
        Logbook method with extra info in text string>
        """
        e1 = "Iterations: "+str(objective_func.niter)+"\n"
        e2 = "Trim delay: "+str(trim_delay)+"\n"
        e3 = "Points Requested: "+str(numPulse)+"\n"
        e5 = "Normalization Amp Coeff: "+str(norm_amp_coeff)+"\n"
        e6 = "Using Live Simplex Seed: "+str(SeedScanBool)+"\n"
        e7 = "Type of optimization: "+(name_opt)+"\n"

        extra_log_text = e1+e2+e3+e5+e6+e7
        self.logbook(objective_func_pv, objective_func, winID, extra_log_text)



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
        device_names = [dev.eid for dev in devices]
        print 'device_names = ',device_names
        self.data['pv_list'] = device_names
        print 'self.data[pv_list] = ',self.data['pv_list']
        for dev in devices:
            self.data[dev.eid] = []
        #print('obj times', objective_func.times)
        for dev in devices:
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
        #data_new = self.recordData(objective_func_pv, objective_func, devices)
        self.recordData(objective_func_pv, objective_func, devices)
        #get the first and last points for GDET gain
        self.detValStart = self.data[objective_func_pv][0]
        self.detValStop  = self.data[objective_func_pv][-1]

        #replace with matlab friendly strings
        for key in self.data:
            key2 = key.replace(":","_")
            self.data[key2] = self.data.pop(key)

        #extra into to add into the save file
        self.data["BEND_DMP1_400_BDES"]   = self.get("BEND:DMP1:400:BDES")
        self.data["Energy"]   = self.get_energy()
        self.data["ScanAlgorithm"]        = str(name_opt)      #string of the algorithm name
        self.data["ObjFuncPv"]            = str(objective_func_pv) #string identifing obj func pv
        self.data["NormAmpCoeff"]         = norm_amp_coeff

        #save data
        import matlog
        self.last_filename=matlog.save("OcelotScan",self.data,path='default')#self.save_path)
        
        print 'Saved scan data to ', self.last_filename
