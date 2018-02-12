#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
--------------------
NOTE: Please configure your editor to use 4 spaces instead of tabs! Spyder does
      this by default.
--------------------

Ocelot GUI, interface for running and testing accelerator optimization methods

This file primarily contains the code for the UI and GUI
The scanner classes are contained in an external file, scannerThreads.py
The resetpanel widget is also contained in a separate module, resetpanel

Tyler Cope, 2016
"""

#QT imports
from __future__ import print_function
from PyQt5.QtWidgets import QApplication, QFrame, QGraphicsWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QObject
from PyQt5 import QtGui, QtCore, Qt, QtWidgets

#normal imports
import numpy as np
import sys
import os
import time
import pyqtgraph as pg
import pandas as pd

#Ocelot files
from mint.lcls_interface import LCLSMachineInterface
from mint.mint import OptControl, Optimizer, Action, GaussProcess, Simplex, RCDS, ESMin#, GaussProcessSKLearn

#local imports
from mint.opt_objects import *
from mint import opt_objects as obj
from sint.corrplot_interface import CorrplotInterface # simulation interface for corrplots
from sint.multinormal_interface import MultinormalInterface # simulation interface for arbitrary dimensions

#for command line options
import argparse

#GUI layout file
from UIOcelotInterface import Ui_Form

# MOVED TO mint/mint.py
#slac python toolbox imports
#try:
#    import matlog
#except:
#    print('No Module named matlog (LCLS e-log)')

# logging
logStdout = False 
if logStdout:
    try: # if running under a profile, save to profile directory
        #username = os.environ['PHYSICS_USER']
        #if username == 'none':
            #username = 'Ocelot'
        #basepath = '/home/physics/' + username + '/OcelotLogs/'

        # save to a directory under the user's home directory
        homepath = os.environ['HOME']
        basepath = homepath + '/ocelot/logs/'
    except:
        basepath = os.environ['PWD']

    try:
        os.makedirs(basepath) # make it if it doesn't exist
    except:
        pass

    #errpath = basepath + 'OcelotLog-stderr-' + time.strftime("%Y_%m_%d_%H_%M") + '.txt'
    #print 'Saving standard error to file ', errpath # notify about path
    #sys.sterr = open(errpath,'w') # redirect all stdout to file

    #inpath = basepath + 'OcelotLog-stdin-' + time.strftime("%Y_%m_%d_%H_%M") + '.txt'
    #print 'Saving standard input to file ', inpath # notify about path
    #sys.sterr = open(inpath,'w') # redirect all stdout to file

    logpath = basepath + 'OcelotLog-stdout-' + time.strftime("%Y_%m_%d_%H_%M") + '.txt'
    print('Saving standard output to file ', logpath) # notify about path
    sys.stdout = open(logpath,'w') # redirect all stdout to file


class OcelotInterfaceWindow(QFrame):
    """ Main class for the GUI application """
    def __init__(self):
        """
        Initialize the GUI and QT UI aspects of the application.
        Initialize the scan parameters.
        Connect start and logbook buttons on the scan panel.
        Initialize the plotting.
        Make the timer object that updates GUI on clock cycle durring a scan.
        """
        # initialize
        QFrame.__init__(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #Make OcelotInterface parent of resetpanel widget
        self.ui.widget.set_parent(self)

        # Flag to allow child modules to query simulation environment
        self.simQ = False

        #Check for devmode --s from command line, set proper mi.
        self.devmodeCheck()

        #method to get defaults for all the scan parameters
        self.setScanParameters()

        #Instantiate machine specific methods
        self.opt_control = OptControl()

        #Clear out callback PV
        self.callbackPV = None

        #Set GP options box disabled
        self.ui.groupBox_2.setEnabled(False)

        #scan panel button connections
        self.ui.startButton.clicked.connect(self.startScan)

        #logbooking
        self.ui.logButton.clicked.connect(lambda:self.logTextVerbose())

        #dropdown for scan device sets
        self.ui.deviceList.activated.connect(self.selectQuads)

        #dropdown to select optimizer
        self.ui.select_optimizer.activated.connect(self.scanMethodSelect)

        #clear table button
        self.ui.remDevice.clicked.connect(self.emptyTable)

        #add device from dropdown
        self.ui.addDevice.clicked.connect(self.addList)

        #launch heatmap button
        self.ui.heatmapButton.clicked.connect(self.launchHeatMap)

        #help and documentation launch button
        self.ui.helpButton.clicked.connect(lambda: os.system("firefox file:///usr/local/lcls/tools/python/toolbox/OcelotInterface/docs/build/html/index.html"))
        #ocelot edm panel for development
        self.ui.devButton.clicked.connect(lambda: os.system("edm -x /home/physics/tcope/edm/ocelot_dev.edl &"))

        #Save path for data, default will put the data in the current matlab data directory
        #See data logging module 'matlog'
        self.save_path = 'default'

        #init plots
        self.addPlots()

        #object funciton selectinator (gdet)
        self.setObFunc()

        #load in the dark theme style sheet
        self.loadStyleSheet()

        #timer for plots, starts when scan starts
        self.multiPvTimer = QtCore.QTimer()
        self.multiPvTimer.timeout.connect(self.getPlotData)

        #Index of optimizer selection dropdown
        self.indexOpt = 0

        #Index of device list dropdown
        self.index = 0

    def loadStyleSheet(self):
        """ Sets the dark GUI theme from a css file."""
        try:
            self.cssfile = "style.css"
            with open(self.cssfile,"r") as f:
                self.setStyleSheet(f.read())
        except IOError:
            print('No style sheet found!')

    def emptyTable(self):
        """ Calls resetpanelbox class method to empty devices from list."""
        self.ui.widget.clearTable()

    def addList(self):
        """ Calls resetpanelbox class method to add devices from list (e.g. LI26 Quads)."""
        self.ui.widget.addTable(self.index)

    def selectQuads(self):
        """ Selects devices to add to table from dropdown, sets index."""
        self.index= self.ui.deviceList.currentIndex()

    def setObFunc(self):
        """
        Method to select new objective function PV (GDET).

        Typically the gas detector, but it could be some other calc PV.
        """
        text = str(self.ui.obj_func_edit.text())
        if text == '':
            self.ui.obj_func_edit.setStyleSheet("color: red")
            return
        state = self.mi.get(str(text))
        print(state)

        if state != None:
            self.objective_func_pv = text
            self.ui.obj_func_edit.setStyleSheet("color: rgb(85, 255, 0);")
            self.plot1.setLabel('left',text=text)
        else:
            self.ui.obj_func_edit.setStyleSheet("color: red")

    def create_devices(self, pvs):
        """
        Method to create devices using only channels (PVs)
        :param pvs: str, device address/channel/PV
        :return: list of the devices [mint.opt_objects.Device(eid=pv[0]), mint.opt_objects.Device(eid=pv[1]), ... ]
        """
        devices = []
        vals = []
        for pv in pvs:
            dev = obj.Device(eid=pv)
            dev.mi = self.mi
            val = dev.get_value()
            devices.append(dev)
            vals.append(val)
        return devices

#==============================================================#
# -------------- Start code simulation/dev UI ---------------- #
#==============================================================#


    def simDisable(self):
        """
        Method to disable development functionality durring production use.
        """
        #objective function edit
        self.ui.obj_func_edit.setEnabled(False)

        #Trim Delay
        self.ui.trim_delay_edit.setEnabled(False)

        #Add Remove/Devices UI Objects
        self.ui.addDevice.setEnabled(False)
        self.ui.remDevice.setEnabled(False)
        self.ui.deviceList.setEnabled(False)
        self.ui.deviceEnter.setEnabled(False)


    def reinitSim(self):
        # python functions only take static init values and Qt slots don't take arguments so need this fcn

        # disconnect slots
        self.ui.lineEdit_numFeat.returnPressed.disconnect(self.reinitSim)
        self.ui.tableWidget_corrMat.itemChanged.disconnect(self.symmetrize)
        self.ui.tableWidget_widthOffset.itemChanged.disconnect(self.updateSimMoments)
        self.ui.pushButton_clearCorr.released.disconnect(self.clearCorrelations)

        # reinit the sim
        self.devmodeCheck(int(self.ui.lineEdit_numFeat.text()))

    def updateObjectivePeak(self):
        # get item value
        try: itemVal = float(self.ui.lineEdit_objPeak.text())
        except:
            itemVal = 1. # default offset

        # update sim
        self.mi.sigAmp = itemVal

        # print to screen
        print("Multinormal simulation objective set to ", self.mi.sigAmp)


    # changing default value of arg ndim below changes the number of sim_devices added
    # ndim=0 uses the corrplot simulation mode
    def devmodeCheck(self, ndim=10):
        #Check for sim mode
        parser = argparse.ArgumentParser(description = 'To launch sim mode')
        parser.add_argument('-s','--s','--sim','--simulation', action = 'store_true', help = "run in simulation mode")
        self.args = parser.parse_args()

        self.sim_ndim = ndim # toggle for corrplot sim (ndim==0) or ndim multinormal sim (ndim>1)

        #setup development mode if devmode==True
        if self.args.s:
            self.simDisable()
            self.simQ = True # flag declaring we're in a simulation mode (so child modules can learn this)

            # change number of features - NOT FULLY IMPLEMENTED
            #self.ui.lineEdit_numFeat.setEnabled(True)
            #self.ui.lineEdit_numFeat.returnPressed.connect(self.reinitSim)

            if ndim == 0:
                #_____________________
                # correlation plot simulation interface

                #corrplotpath = '/u1/lcls/matlab/data/2016/2016-10/2016-10-20/CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat'
                #corrplotpath = '/u1/lcls/matlab/data/2016/2016-10/2016-10-20/CorrelationPlot-QUAD_LI26_801_BCTRL-2016-10-20-055153.mat'
                #corrplotpath = '/u1/lcls/matlab/data/2018/2018-01/2018-01-23/CorrelationPlot-QUAD_LTU1_620_BCTRL-2018-01-23-175201.mat'
                #corrplotpath = 'sint/corrplots/CorrelationPlot-QUAD_IN20_511_BCTRL-2016-10-20-053315.mat'
                #corrplotpath = 'sint/corrplots/CorrelationPlot-QUAD_LI26_801_BCTRL-2016-10-20-055153.mat'
                corrplotpath = 'sint/corrplots/CorrelationPlot-QUAD_LTU1_620_BCTRL-2018-01-23-175201.mat'
                
                print('Trying CorrplotInterface with ', corrplotpath)
                self.mi = CorrplotInterface(corrplotpath)
                try:
                    print('Trying CorrplotInterface with ', corrplotpath)
                    self.mi = CorrplotInterface(corrplotpath)
                except:
                    ndim = 2 # probably couldn't find corrplotpath so launch
                             # the MultinormalSim mode with 2 quads
                    print('\nWARNING: Could not launch CorrplotInterface.\n')
                    print('INFO:    Trying MultinormalInterface with ',ndim,' devices.\n')

            if ndim != 0:

                #_____________________
                # multivariate normal simulation interface

                # ---- start sim settings ----

                # declare simulation params
                #ndim = 8 # number of devices (default set by argument of fcn)
                #sigAmp = 1. # objective peak
                bgNoise = 0.064 # something typical from old corrplt data
                sigNoiseScaleFactor = 0.109 # seems like something typical is amp_noise / sqrt(amp_signal) ~= 0.193/np.sqrt(3.113) = 0.109
                noiseScaleFactor = 1. # easy to use this as a noise toggle: 0 turns off noise; 1 turns it on (default)

                # these set the statistical properties of the 
                offset_nsigma = 2. # scales the magnitude of the distance between start and goal so that the distance has a zscore of nsigma
                offsets = np.random.randn(ndim) #1.*np.ones(ndim) # peak location is an array
                offsets = np.round(offsets*offset_nsigma/np.linalg.norm(offsets),2) #1.*np.ones(ndim) # peak location is an array
                projected_widths = np.ones(ndim) # widths of the marginalized distributions
                correlation_matrix = np.diag(np.ones(ndim)) # correlations between coordinates
                self.GP_hyp_file = "devmode" # this preloads some hyper params for GP instead of loading paramters/hype3.npy which doesn't have our sim PVs

                # simulation objective function amplitude
                self.ui.lineEdit_objPeak.setEnabled(True)
                self.ui.lineEdit_objPeak.returnPressed.connect(self.updateObjectivePeak)

                # ---- end sim settings ----

                # import interface
                self.mi = MultinormalInterface(offsets, projected_widths, correlation_matrix)
                #sint.sigAmp = sigAmp
                self.updateObjectivePeak()
                self.mi.bgNoise = bgNoise
                self.mi.sigNoiseScaleFactor = sigNoiseScaleFactor
                self.mi.noiseScaleFactor = noiseScaleFactor
                #interface.noiseScaleFactor = 1. # control the noise
                #interface.numSamples = nSamplesPerPoint
                #interface.SNRgoal = SNRgoal
                #interface.numBatchSamples = numBatchSamples
                #interface.maxNumSamples = maxNumSamples

                # number of features
                self.ui.lineEdit_numFeat.setText(str(ndim))

                # setup correlation matrix table
                self.ui.tableWidget_corrMat.setRowCount(len(self.mi.corrmat))
                self.ui.tableWidget_corrMat.setColumnCount(len(self.mi.corrmat[0]))
                for i,row in enumerate(self.mi.corrmat):
                    for j,val in enumerate(row):
                        self.ui.tableWidget_corrMat.setItem(i,j,QtWidgets.QTableWidgetItem(str(val)))
                # for later resizing
                self.ui.tableWidget_corrMat.maxWidth = 2000
                self.ui.tableWidget_corrMat.maxHeight = 2000
                #self.ui.tableWidget_corrMat.maxWidth = self.ui.tableWidget_corrMat.width()
                #self.ui.tableWidget_corrMat.maxHeight = self.ui.tableWidget_corrMat.height()
                # resize cells to fit data
                self.ui.tableWidget_corrMat.resizeColumnsToContents()
                self.ui.tableWidget_corrMat.resizeRowsToContents()
                # resize tableWidget container to fit table
                self.resizeTableWidget(self.ui.tableWidget_corrMat)
                # connect actions to slots
                self.ui.tableWidget_corrMat.itemChanged.connect(self.symmetrize)
                self.ui.pushButton_clearCorr.released.connect(self.clearCorrelations)

                # setup widths and centroids table
                self.ui.tableWidget_widthOffset.setRowCount(len(self.mi.offsets))
                self.ui.tableWidget_widthOffset.setColumnCount(2)
                for i,val in enumerate(self.mi.offsets):
                    self.ui.tableWidget_widthOffset.setItem(i,0,QtWidgets.QTableWidgetItem(str(val)))
                for i,val in enumerate(self.mi.sigmas):
                    self.ui.tableWidget_widthOffset.setItem(i,1,QtWidgets.QTableWidgetItem(str(val)))
                # for later resizing
                self.ui.tableWidget_widthOffset.maxWidth = 2000
                self.ui.tableWidget_widthOffset.maxHeight = 2000
                #self.ui.tableWidget_widthOffset.maxWidth = self.ui.tableWidget_widthOffset.width()
                #self.ui.tableWidget_widthOffset.maxHeight = self.ui.tableWidget_widthOffset.height()
                # resize cells to fit data
                self.ui.tableWidget_widthOffset.resizeColumnsToContents()
                self.ui.tableWidget_widthOffset.resizeRowsToContents()
                # resize tableWidget container to fit table
                self.resizeTableWidget(self.ui.tableWidget_widthOffset)
                # connect actions to slots
                self.ui.tableWidget_widthOffset.itemChanged.connect(self.updateSimMoments)

            # end simulation setup if/else

            print("Using Simulation Interface")

        else:
            self.ui.tabWidget.setTabEnabled(2, False)
            self.mi = LCLSMachineInterface()
            print("Launching Ocelot in Normal Mode for LCLS")


    # resize tableWidget container to fit simulation correlation table
    def updateSimMoments(self):
        tableWidget = self.ui.tableWidget_widthOffset
        tableWidget.itemChanged.disconnect(self.updateSimMoments)

        # selectedItems()
        #for item in self.tableView.selectedItems():
            #print "selectedItems", item.text()

        #print 'self.mi.offsets = ', self.mi.offsets
        #print 'self.mi.sigmas = ', self.mi.sigmas
        #print 'self.mi.invcovarmat = ', self.mi.invcovarmat

        # selectedIndexes()
        for item in tableWidget.selectedIndexes():
            # print "selectedIndexes", item.row(), item.column()

            # get item value
            itemVal = tableWidget.item(item.row(), item.column()).text()
            try: itemVal = float(tableWidget.item(item.row(), item.column()).text())
            except:
                if item.column()==0: itemVal = 0. # default offset
                else: itemVal = 1. # default width
            if item.column()==1: # special cases for width values
                if itemVal == 0: itemVal = 1.
                if itemVal < 0: itemVal = abs(itemVal)

            # update table
            tableWidget.setItem(item.row(), item.column(), QtWidgets.QTableWidgetItem(str(itemVal)))

            # update simulation moments
            if item.column()==0:
                self.mi.offsets[item.row()] = itemVal
            if item.column()==1:
                self.mi.sigmas[item.row()] = itemVal

        # update simulation correlation matrix
        self.mi.store_moments(self.mi.offsets, self.mi.sigmas, self.mi.corrmat)
        #print 'self.mi.offsets = ', self.mi.offsets
        #print 'self.mi.sigmas = ', self.mi.sigmas
        #print 'self.mi.invcovarmat = ', self.mi.invcovarmat

        #print self.tableView.item(item.row(), item.column())
        #print self.tableView.rowCount()
        self.resizeTableWidget(tableWidget)
        tableWidget.itemChanged.connect(self.updateSimMoments)


    # resize tableWidget container to fit simulation correlation table
    def resizeTableWidget(self, tableWidget, redo = True):
        # resize stuff to fit
#        tableWidget.resizeColumnsToContents()

        # horizontal
        tableWidget.maxWidth = max([tableWidget.width(), tableWidget.maxWidth])
        width = tableWidget.verticalHeader().width()
        width += tableWidget.horizontalHeader().length()
        if tableWidget.verticalScrollBar().isVisible():
            width += tableWidget.verticalScpushButton_clearCorrrollBar().width()
        width += tableWidget.frameWidth() * 2
        tableWidget.setFixedWidth(min([width,tableWidget.maxWidth]))

        # vertical
        tableWidget.maxHeight = max([tableWidget.height(), tableWidget.maxHeight])
        height = tableWidget.verticalHeader().length()
        height += tableWidget.horizontalHeader().height()
        if tableWidget.horizontalScrollBar().isVisible():
            height += tableWidget.horizontalScrollBar().height()
        height += tableWidget.frameWidth() * 2
        tableWidget.setFixedHeight(min([height,tableWidget.maxHeight]))

        # apparently we may have to call this fcn twice due to a glitch in scrollbar isVisible
        if redo: self.resizeTableWidget(tableWidget, False)


    # symmetrize the simulation correlation table
    def symmetrize(self):
        self.ui.tableWidget_corrMat.itemChanged.disconnect(self.symmetrize)

        # selectedItems()
        #for item in self.tableView.selectedItems():
            #print "selectedItems", item.text()

        # print 'self.mi.corrmat = ', self.mi.corrmat
        # print 'self.mi.invcovarmat = ', self.mi.invcovarmat

        # selectedIndexes()
        for item in self.ui.tableWidget_corrMat.selectedIndexes():
            # print "selectedIndexes", item.row(), item.column()

            # get item value
            itemVal = self.ui.tableWidget_corrMat.item(item.row(), item.column()).text()
            try: itemVal = float(self.ui.tableWidget_corrMat.item(item.row(), item.column()).text())
            except: itemVal = 0.
            # print "itemVal = ", itemVal
            if abs(itemVal) > 0.99: itemVal = 0.99 * np.sign(itemVal)
            # print "itemVal = ", itemVal
            if item.row() == item.column(): itemVal = 1.
            # print "itemVal = ", itemVal

            # update table
            self.ui.tableWidget_corrMat.setItem(item.row(), item.column(), QtWidgets.QTableWidgetItem(str(itemVal)))
            self.ui.tableWidget_corrMat.setItem(item.column(), item.row(), QtWidgets.QTableWidgetItem(str(itemVal)))

            # update simulation correlation matrix
            self.mi.corrmat[item.row(), item.column()] = itemVal
            self.mi.corrmat[item.column(), item.row()] = itemVal

        # update simulation correlation matrix
        self.mi.store_moments(self.mi.offsets, self.mi.sigmas, self.mi.corrmat)
        # print 'self.mi.corrmat = ', self.mi.corrmat
        # print 'self.mi.invcovarmat = ', self.mi.invcovarmat

        #print self.tableView.item(item.row(), item.column())
        #print self.tableView.rowCount()
        # resize stuff to fit
        self.ui.tableWidget_corrMat.resizeColumnsToContents()
        self.resizeTableWidget(self.ui.tableWidget_corrMat)
        self.ui.tableWidget_corrMat.itemChanged.connect(self.symmetrize)


    # clear off-diagonal elements in the correlation matrix
    def clearCorrelations(self):
        self.ui.tableWidget_corrMat.itemChanged.disconnect(self.symmetrize)
        # unit matrix
        self.mi.corrmat = np.diag(np.ones(self.sim_ndim))

        # update table
        for i,row in enumerate(self.mi.corrmat):
            for j,val in enumerate(row):
                self.ui.tableWidget_corrMat.setItem(i,j,QtWidgets.QTableWidgetItem(str(val)))

        self.resizeTableWidget(self.ui.tableWidget_corrMat)
        self.ui.tableWidget_corrMat.itemChanged.connect(self.symmetrize)


    def devmode(self):
        """
        Used to setup a development mode for quick testing.

        This method contains settings for a dev mode on GUI startup.

        Loads data from correlation plot scan.
        """
        #faster timing
        self.trim_delay = 1.e-6 #fast trim time
        self.data_delay = 1.e-6 #fast delay time

        #print "WARNING: Slowing the trim_delay to 1 second!!"
        #self.trim_delay = 1.

        #GP settings
        if(hasattr(self.mi, 'offsets')):
            self.GP_hyp_file = "devmode"
        else:
            self.GP_hyp_file = "parameters/hype3.npy"
        #self.seedScanBool = False
        #set the save path to tmp instead of the lcls matlab data directory
        self.save_path = '/tmp/'
        self.ui.widget.devices = []
        self.pvs = []
        # DUPLICATE EFFORT IN FCN main BELOW
        for dev in self.mi.pvs[:-1]: # add ctrl PVs (all but last in sim interface)
            self.pvs.append(str(dev))
            self.ui.widget.addPv(dev)
            #print dev
        #self.ui.widget.addPv(str(self.pvs[0]))
        #self.ui.widget.addPv(self.pvs[1])
        self.devices = []
        self.devices = self.ui.widget.get_devices(self.pvs[:-1])
        self.objective_func_pv = self.mi.pvs[-1]
        self.scanMethodSelect()

#==============================================================#
# -------------- Start code for scan options UI -------------- #
#==============================================================#

    def setScanParameters(self):
        """
        Initialize default parameters for a scan when the GUI starts up.

        Creates connection for parameter changes on options panel.
        """
        #normalization amp coeff for scipy scanner and GP hyps
        self.norm_amp_coeff = 1.0
        self.ui.norm_scale_edit.setText(str(self.norm_amp_coeff))
        self.ui.norm_scale_edit.returnPressed.connect(self.setNormAmpCoeff)

        #set objection method (gdet or some other pv to optimize)
        if self.args.s:
            self.objective_func_pv = str(self.mi.pvs[-1])
        else:
            self.objective_func_pv = "GDET:FEE1:241:ENRCHSTBR"
        self.ui.obj_func_edit.setText(str(self.objective_func_pv))
        self.ui.obj_func_edit.returnPressed.connect(self.setObFunc)

        #For manually adding device to scan by typing it in
        self.ui.deviceEnter.returnPressed.connect(self.setDevice)

        #set trim delay
        self.trim_delay = 1.0
        self.ui.trim_delay_edit.setText(str(self.trim_delay))
        self.ui.trim_delay_edit.returnPressed.connect(self.setTrimDelay)

        #set data delay
        self.numPulse = 120
        self.ui.data_points_edit.setText(str(self.numPulse))
        self.ui.data_points_edit.returnPressed.connect(self.setPoints)

        #set GP Seed data file
        self.GP_seed_file = "parameters/simSeed.mat"
        self.ui.seed_file_edit.setText(str(self.GP_seed_file))
        self.ui.seed_file_edit.returnPressed.connect(self.setGpSeed)

        #set GP Hyperparameters from a file
        # NOTE TO ADAM: yes, it's messy and not the best place to do this,
        #               BUT setScanParameters is called after devmodeCheck
        #               so self.GP_hyp_file is overwritten otherwise
        #               Feel free to fix in a better way but preserve this override.
        if(self.devmode and hasattr(self.mi, 'offsets')):
            self.GP_hyp_file = "devmode"
        else:
            self.GP_hyp_file = "parameters/hype3.npy"

        self.ui.hyps_edit.setText(str(self.GP_hyp_file))
        self.ui.hyps_edit.returnPressed.connect(self.setGpHyps)

        #To toggle between using simplex scaling or GP hyper scaling
        self.useNormScale = True

        #set the "use GP Simplex Seed" bool for the GP optimizer class
        #self.seedScanBool = False
        self.ui.live_seed_check.stateChanged.connect(self.setGpSimplexSeed)
        self.ui.live_seed_check.setCheckState(2)

        #initialize algorithm names for UI, add items to combobox
        self.name1 = "Nelder-Mead Simplex"
        self.name2 = "Gaussian Process"
        self.name3 = "scikit-learn Gaussian Process"
        self.name4 = "RCDS"
        self.name5 = "ES minimizer"
        self.ui.select_optimizer.addItem(self.name1)
        self.ui.select_optimizer.addItem(self.name2)
        #self.ui.select_optimizer.addItem(self.name3)
        #self.ui.select_optimizer.addItem(self.name4)
        #self.ui.select_optimizer.addItem(self.name5)

        #initialize GUI with simplex method
        self.name_opt = "Nelder-Mead Simplex"

        # choose objective function statistic to tune on
        self.stat_names = ['Mean','Median','Standard deviation','Median deviation','Max','Min','80th percentile','average of points > mean','20th percentile']
        for name in self.stat_names:
            self.ui.select_statistic.addItem(name)

    def setDevice(self):
        """
        Method to select new Device to Scan, must be valid PV.

        """
        text = str(self.ui.deviceEnter.text())
        #check for blank string that will break it
        if text == '':
            self.ui.deviceEnter.setStyleSheet("color: red")
            return #exit

        state = self.mi.get(str(text))
        print(state)

        if state != None:
            self.ui.widget.addPv(text)
            self.ui.deviceEnter.clear()

    def setNormAmpCoeff(self):
        """Changes the scaling parameter for the simplex/scipy normalization."""
        try:
            self.norm_amp_coeff = float(self.ui.norm_scale_edit.text())
            print("Norm scaling coeff = ", self.norm_amp_coeff)
        except:
            self.ui.norm_scale_edit.setText(str(self.norm_amp_coeff))
            print("Bad float for norm amp coeff")

    def setTrimDelay(self):
        """
        Select a new trim time for a device from GUI line edit.

        Scanner will wait this long before starting data acquisition.
        """
        try:
            self.trim_delay = float(self.ui.trim_delay_edit.text())
            print("Trim delay =",self.trim_delay)
        except:
            self.ui.trim_delay.setText(str(self.trim_delay))
            print("bad float for trim delay")

    def setPoints(self):
        """
        Select Number of points to average if objective function is a waveform

        SLACTarget object determines the time to wait before collecting data
        """
        try:
            self.numPulse = float(self.ui.data_points_edit.text())
            print("Number of Points to average =",self.numPulse)
        except:
            self.ui.data_points_edit.setText(str(self.numPulse))
            print("bad float for data delay")

    def setGpSeed(self):
        """
        Set directory string to use for the GP scanner seed file.
        """
        self.GP_seed_file = str(self.ui.seed_file_edit.text())

    def setGpHyps(self):
        """
        Set directory string to use for the GP hyper parameters file.
        """
        self.GP_hyp_file = str(self.ui.hyps_edit.text())
        print(self.GP_hyp_file)

    def setGpSimplexSeed(self):
        """
        Sets the bool to run GP in a simplex seed mode.
        """
        if self.ui.live_seed_check.isChecked():
            self.seedScanBool = True
        else:
            self.seedScanBool = False
        try:
            self.minimizer.seedScanBool = self.seedScanBool
            print("GP seed bool == ",self.seedScanBool)
        except:
            print("Unable to toggle the simplex seed flag. Try selecting a different optimizer.")

#========================================================================#
# -------------- Start code for running optimization scan -------------- #
#========================================================================#

    def scanMethodSelect(self):
        """
        Sets scanner method from options panel combo box selection.  Creates Objective Function object


        This method executes from the startScan() method, when the UI "Start Scan" button is pressed.

        Returns:
                 Selected scanner object
        """
        self.indexOpt = self.ui.select_optimizer.currentIndex()
        self.name_opt = self.ui.select_optimizer.currentText()
        self.objective_func = obj.SLACTarget(eid=self.objective_func_pv)
        self.objective_func.points = self.numPulse
        self.objective_func.mi = self.mi
        self.data_delay= self.mi.dataDelay(self.objective_func_pv, self.numPulse)
        self.stat_name = self.ui.select_statistic.currentText() # discover statistic
        self.objective_func.mi.stat_name = self.stat_name

        #simplex Method
        indexOpt = 0
        if self.indexOpt == indexOpt:
            self.minimizer = Simplex()
            self.ui.groupBox_2.setEnabled(False)
            self.useNormScale = True

        #GP Method
        indexOpt += 1
        if self.indexOpt == indexOpt:
            self.minimizer = GaussProcess()
            self.minimizer.seed_timeout = self.trim_delay+self.data_delay
            self.ui.groupBox_2.setEnabled(True)
            self.useNormScale = False
            self.minimizer.seedScanBool = self.seedScanBool
            self.minimizer.mi = self.mi

        ## scikit-learn GP
        #indexOpt += 1
        #if self.indexOpt == indexOpt:
            #self.minimizer = GaussProcessSKLearn()
            #self.minimizer.seed_timeout = self.trim_delay+self.data_delay
            #self.ui.groupBox_2.setEnabled(True)
            #self.useNormScale = False

        ## RCDS (augmented Powell's Method)
        #indexOpt += 1
        #if self.indexOpt == indexOpt:
            #self.name_current = self.name4
            #self.minimizer = RCDS()
            #self.ui.groupBox_2.setEnabled(False)
            #self.useNormScale = False

        ## Extremum seeking method
        #indexOpt += 1
        #if self.indexOpt == indexOpt:
            #self.name_current = self.name5
            #self.minimizer = ESMin()
            #self.ui.groupBox_2.setEnabled(False)
            #self.useNormScale = False

        print("Selected Algorithm =", self.name_opt)
        return self.minimizer

    def closeEvent(self, event):
        """ Happens upon user exit."""
        if self.ui.startButton.text() == "Stop Scan":
            self.opt.opt_ctrl.stop()
            del(self.opt)
            self.ui.startButton.setStyleSheet("color: rgb(85, 255, 127);")
            self.ui.startButton.setText("Start Scan")
            return 0
        QFrame.closeEvent(self, event)

    def setInputs(self):
        """ Sets all the user inputs from UI """
        #self.data_delay= self.mi.dataDelay(self.objective_func_pv, self.numPulse)
        self.objective_func.points = self.numPulse
        self.opt.multiplier = self.norm_amp_coeff
        self.opt.points = self.numPulse
        self.minimizer.multiplier = self.norm_amp_coeff
        self.opt.normalization = self.useNormScale
        self.opt.timeout    = self.trim_delay+self.data_delay
        self.minimizer.max_iter = 50
        #print ' WARNING self.minimizer.max_iter = 5'
        self.minimizer.hyper_file = "parameters/hype3.npy"
        #self.minimizer.hyper_file = "parameters/fit_params.pkl"

    def startScan(self):
        """
        This starts the optimizer thread and sets the algorithm as the "minimizer"

        This method executes when the UI "Start Scan" button is pressed.

        """
        print("Start Scan Clicked...")
        self.mi.setListener((self.indexOpt + 1))
        self.scanStartTime = time.time()
        self.pvs = self.ui.widget.getPvsFromCbState()
        self.devices = self.ui.widget.get_devices(self.pvs)
        if self.ui.startButton.text() == "Stop Scan":
            self.opt.opt_ctrl.stop()
            self.finishScript()
            return 0
        print(("Multi Plot setup for :", self.pvs))
        self.setUpMultiPlot(self.pvs)
        self.scanMethodSelect()
        self.opt = Optimizer()
        self.opt.minimizer = self.minimizer
        self.setInputs()
        self.opt.seq = [Action(func=self.opt.max_target_func, args=[self.objective_func, self.devices])]
        self.opt.start()
        print("Starting Timer for Plot")
        self.multiPvTimer.start(100)
        self.ui.startButton.setText("Stop Scan")
        self.ui.startButton.setStyleSheet("color:red")

    def scanFinished(self, guimode = True):
        """
        Reset the GUI after a scan is complete.  Runs on QTimer set in Main()
        """
        try:
            if not self.opt.isAlive() and self.ui.startButton.text() == "Stop Scan":
                self.finishScript()
        except:
            pass

    def finishScript(self):
        print("Scan Finished")
        
        #set flag PV to zero
        self.mi.setListener(0)
        del(self.opt)
        
        # save data
        skipSimSave = False
        if self.args.s and skipSimSave:
            pass
        else:
            try:
                print('Saving scan data.')
                
                # first try to gather minimizer data
                try:
                    self.minimizer.saveModel() # need to save GP model first
                except:
                    pass
                    
                # now save machine data: OcelotScan*.mat file
                self.saveData()
                
            except:
                print('\nWARNING: Could not save data.\n')
        
        # now pickle Ocelot objects
        #self.pickleObjects()
        
        #reset UI controls
        print("DEBUG HUGO - Stop Timer Plot")
        self.multiPvTimer.stop()
        self.ui.startButton.setStyleSheet("color: rgb(85, 255, 127);")
        self.ui.startButton.setText("Start scan")

    ##############from GP scanner Threads, not sure if we use this.
    
    # looks like this stuff was moved ...

    #def loadModelParams(self, model, filename):
        #"""
        #Method to build the GP model using loaded model parameters

        #Can give this method an ocelot save file to load in that files model

        #Args:
                #filename (str): String for the file directory
                #model (object): Takes in a GP model object

        #Returns:
                #GP model object with parameters from loaded data
        #"""
        #model_file = scipy.io.loadmat(filename)['data']

        #model.alpha        = model_file['alpha'].flatten(0)[0]
        #model.C            = model_file['C'].flatten(0)[0]
        #model.BV           = model_file['BV'].flatten(0)[0]
        #model.covar_params = model_file['covar_params'].flatten(0)[0]
        #model.covar_params = (model.covar_params[0][0],model.covar_params[1][0])
        #model.KB           = model_file['KB'].flatten(0)[0]
        #model.KBinv        = model_file['KBinv'].flatten(0)[0]
        #model.weighted     = model_file['weighted'].flatten(0)[0]

        #print
        #print 'Loading in new model from:'
        #print filename
        #print

        #return model

    #def saveModel(self):
        #"""
        #Add GP model parameters to the save file.
        #"""
        ##add in extra GP model data to save
        #self.mi.data["alpha"]        = self.model.alpha
        #self.mi.data["C"]            = self.model.C
        #self.mi.data["BV"]           = self.model.BV
        #self.mi.data["covar_params"] = self.model.covar_params
        #self.mi.data["KB"]           = self.model.KB
        #self.mi.data["KBinv"]        = self.model.KBinv
        #self.mi.data["weighted"]     = self.model.weighted
        #self.mi.data["noise_var"]    = self.model.noise_var
        #self.mi.data["pv_list"]      = self.pvs

#======================================================================#
# -------------- Start code for setting/updating plots --------------- #
#======================================================================#

    def getPlotData(self):
        """
        Collects data and updates plot on every GUI clock cycle.
        """
        #get x,y obj func data from the machine interface
        print("********************* DEBUG HUGO - getPlotData")
        try:
            print("Will fetch values...")
            y = self.objective_func.values
            print("DEBUG HUGO: Values: ", y)
        except:
            self.scanFinished
        try:
            print("obj_func: ",self.objective_func.times) 
            x = np.array(self.objective_func.times) - self.objective_func.times[0]
            print("DEBUG HUGO - set data...")
            #set data to like pg line object
            self.obj_func_line.setData(x=x,y=y)

            for dev in self.devices:
                print(("DEBUG HUGO - set data for dev: ", dev))
                y = np.array(dev.values)-self.multiPlotStarts[dev.eid]
                x = np.array(dev.times) - np.array(dev.times)[0]
                line = self.multilines[dev.eid]
                line.setData(x=x, y=y)
        except Exception as e:
            print('No data to plot yet', e)


    def addPlots(self):
        """
        Initializes the GUIs plots and labels on startup.
        """
        #setup plot 1 for obj func monitor
        self.plot1 = pg.PlotWidget(parent=self, title = "Objective Function Monitor",labels={'left':str(self.objective_func_pv),'bottom':"Time (seconds)"})
        self.plot1.showGrid(1,1,1)
        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes
        layout = QtWidgets.QGridLayout()
        self.ui.widget_2.setLayout(layout)
        layout.addWidget(self.plot1,0,0)

        #setup plot 2 for device monitor
        self.plot2 = pg.PlotWidget(parent=self, title = "Device Monitor",labels={'left':"Device (Current - Start)",'bottom':"Time (seconds)"})
        self.plot2.showGrid(1,1,1)
        self.plot2.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes
        layout = QtWidgets.QGridLayout()
        self.ui.widget_3.setLayout(layout)
        layout.addWidget(self.plot2,0,0)

        #legend for plot 2
        self.leg2 = customLegend(offset=(75,20))
        self.leg2.setParentItem(self.plot2.graphicsItem())

        #create the obj func line object
        color = QtGui.QColor(0,255,255)
        pen=pg.mkPen(color,width=3)
        self.obj_func_line = pg.PlotCurveItem(x=[],y=[],pen=pen,antialias=True)
        print(("Will add Item: ", self.obj_func_line))
        self.plot1.addItem(self.obj_func_line)

    def randColor(self):
        """
        Generate random line color for each device plotted.

        Returns:
                QColor object of a random color
        """
        hi = 255
        lo = 128
        c1 = np.random.randint(lo,hi)
        c2 = np.random.randint(lo,hi)
        c3 = np.random.randint(lo,hi)
        return QtGui.QColor(c1,c2,c3)

    def setUpMultiPlot(self,pvs):
        """
        Reset plots when a new scan is started.
        """
        self.plot2.clear()
        self.multilines      = {}
        self.multiPvData     = {}
        self.multiPlotStarts = {}
        x = []
        y = []
        self.leg2.scene().removeItem(self.leg2)
        self.leg2 = customLegend(offset=(50,10))
        self.leg2.setParentItem(self.plot2.graphicsItem())

        default_colors = [QtGui.QColor(255,51,51),QtGui.QColor(51,255,51),QtGui.QColor(255,255,51),QtGui.QColor(178,102,255)]
        for i in range(len(pvs)):

            #set the first 4 devices to have the same default colors
            if i < 4:
                color = default_colors[i]
            else:
                color = self.randColor()

            pen=pg.mkPen(color,width=2)
            self.multilines[pvs[i]]  = pg.PlotCurveItem(self.plot2, x,y,pen=pen,antialias=True,name=str(pvs[i]))
            self.multiPvData[pvs[i]] = []
            self.multiPlotStarts[pvs[i]] = self.mi.get(pvs[i])
            print(("Will add Item to plot 2: ", self.multilines[pvs[i]]))
            self.plot2.addItem(self.multilines[pvs[i]])
            self.leg2.addItem(self.multilines[pvs[i]],pvs[i],color=str(color.name()))


    def launchHeatMap(self):
        """
        Launches script to display a GP heatmap of two PVs selected from table.

        Can only show data from the GUIs last scan.
        """
        pvnames = self.ui.widget.getPvsFromCbState()
        if len(pvnames) != 2:
            print("Pick only 2 PVs for a slice!")
            return
        com = "python ./GP/analyze_script.py "+str(self.mi.last_filename)+" "+pvnames[0]+" "+pvnames[1]+" &"
        print('Heatmap command:',com)
        os.system(com)

#======================================================================#
# -------------- Start code for saving/logbooking data --------------- #
#======================================================================#

    def saveData(self):
        self.mi.saveData(self.objective_func_pv, self.objective_func, self.devices, self.name_opt, self.norm_amp_coeff)

    def logTextVerbose(self):
        self.mi.logTextVerbose(self.objective_func_pv, self.objective_func, self.trim_delay, self.numPulse, self.norm_amp_coeff, self.seedScanBool, self.name_opt, self.winId())
        
    def pickleObjects(self):
        
        # need to remove stuff that isn't picklable
        # i think i did this in GP/BayesOptimization.py or GP/parallel*.py
        
        # pickling the vars or .__dict__ of an object eliminates need to import later?
        
        # gather objects to pickle
        
        objects = {}
        #objects['mi_vars'] = vars(self.mi)
        #try:
            #objects['minimizer_vars'] = vars(self.minimizer)
        #except:
            #pass
        #try:
            #objects['model_vars'] = vars(self.minimizer.model)
        #except:
            #pass
        #objects['objective_func_vars'] = vars(self.objective_func)
        objects['objective_func_pv'] = self.objective_func_pv
        #objects['devices'] = self.devices
        objects['name_opt'] = self.name_opt
        objects['norm_amp_coeff'] = self.norm_amp_coeff
        objects['trim_delay'] = self.trim_delay
        objects['numPulse'] = self.numPulse
        objects['norm_amp_coeff'] = self.norm_amp_coeff
        objects['seedScanBool'] = self.seedScanBool
        
        # figure out directory to save to
        
        try: # if running under a profile, save to profile directory
            #username = os.environ['PHYSICS_USER']
            #if username == 'none':
                #username = 'Ocelot'
            #basepath = '/home/physics/' + username + '/OcelotObjects/'

            # save to a directory under the user's home directory
            homepath = os.environ['HOME']
            basepath = homepath + '/ocelot/objects/'
        except:
            basepath = os.environ['PWD']

        # make directory
        
        try:
            os.makedirs(basepath) # make it if it doesn't exist
        except:
            pass

        # name pickle

        pklpath = basepath + 'OcelotObjects-' + time.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        
        # pickle
        
        import pickle
        filehandler = open(pklpath, 'w')
        pickle.dump(objects, filehandler) 
        filehandler.close()
        
        print('Pickled Ocelot objects to file ', pklpath) # notify about path
        


#==========================================================================#
# -------------- Start code for reformating the plot legend -------------- #
#==========================================================================#


# Ignore most of thus stuff, only cosmetic for device plot

class customLegend(pg.LegendItem):
    """
    STUFF FOR PG CUSTOM LEGEND (subclassed from pyqtgraph).
    Class responsible for drawing a single item in a LegendItem (sans label).
    This may be subclassed to draw custom graphics in a Legend.
    """
    def __init__(self,size=None,offset=None):
        pg.LegendItem.__init__(self,size,offset)

    def addItem(self, item, name, color="CCFF00"):

        label = pg.LabelItem(name,color=color,size="6pt",bold=True)
        sample = None
        row = self.layout.rowCount()
        self.items.append((sample, label))
        self.layout.addItem(sample, row, 0)
        self.layout.addItem(label, row, 1)
        self.layout.setSpacing(0)

class ItemSample(pg.GraphicsWidget):
    """ MORE STUFF FOR CUSTOM LEGEND """

    ## Todo: make this more generic; let each item decide how it should be represented.
    def __init__(self, item):
        pg.GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        #p.setRenderHint(p.Antialiasing)  # only if the data is antialiased.
        opts = self.item.opts

        if opts.get('fillLevel',None) is not None and opts.get('fillBrush',None) is not None:
            p.setBrush(fn.mkBrush(opts['fillBrush']))
            p.setPen(fn.mkPen(None))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(2,18), QtCore.QPointF(18,2), QtCore.QPointF(18,18)]))

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(2, 18, 18, 2)

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts

            pen = fn.mkPen(opts['pen'])
            brush = fn.mkBrush(opts['brush'])
            size = opts['size']

            p.translate(10,10)
            path = drawSymbol(p, symbol, size, pen, brush)

#==============================================================#
# --------------- main method for starting GUI --------------- #
#==============================================================#

def main():

    """
    Function to start up the main program.

    Development mode:
    If devmode == False - GUI defaults to normal parameter list, defaults to nelder mead simplex
    if devmode == True  - GUI loads old correlation plot scan data, pvs, and objective function.
    """

    pvs = 'parameters/lclsparams' #default filename

    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)

    #create the application
    app    = QApplication(sys.argv)
    window = OcelotInterfaceWindow()

    #timer for end of scan, need to look at new threading methods using QT for Optimizer thread.
    timerFin = pg.QtCore.QTimer()
    timerFin.timeout.connect(window.scanFinished)
    timerFin.start(300)

    #setup development mode if devmode==True
    if window.args.s:
        devmode = True
    else:
        devmode = False
    if devmode:
        pvs = []
        for dev in window.mi.pvs[:-1]: # DUPLICATE EFFORT IN FCN devmode ABOVE
            pvs.append(str(dev))
        window.devmode()
    else:
        pass

    #Build the PV list from dev PVs or selected source
    print(("PVs = ", pvs))
    window.ui.widget.getPvList(pvs)

    #set checkbot status
    if not devmode:
        window.ui.widget.uncheckBoxes()

    #show app
    window.setWindowIcon(QtGui.QIcon('ocelot.png'))
    window.show()

    #Build documentaiton if source files have changed
    os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    sys.exit(app.exec_())

#_________________________
if __name__ == "__main__":
    main()
