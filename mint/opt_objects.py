"""
Objective function and devices
S.Tomin, 2017
"""

import numpy as np
import time



class Device(object):
    def __init__(self, eid=None):
        self.eid = eid
        self.id = eid
        self.values = []
        self.times = []
        self.simplex_step = 0
        self.mi = None
        self.dp = None

    def set_value(self, val):
        """ Set Value for devices, record time and value """
        self.values.append(val)
        self.times.append(time.time())
        self.mi.set_value(self.eid, val)

    def get_value(self):
        """ Standard get_value method for devices """
        val = self.mi.get_value(self.eid)
        return val

    def state(self):
        """
        Check if device is readable

        :return: state, True if readable and False if not
        """
        state = True
        try:
            self.get_value()
        except:
            state = False
        return state

    def clean(self):
        """ Init time and value list for device """
        self.values = []
        self.times = []

    def check_limits(self, value):
        limits = self.get_limits()
        if value < limits[0] or value > limits[1]:
            print('limits exceeded', value, limits[0], value, limits[1])
            return True
        return False

    def get_limits(self):
        if self.dp is not None:
            return self.dp.get_limits(self.eid)
        else:
            return self.mi.get_limits(self.eid)


# for testing
class TestDevice(Device):
    def __init__(self, eid=None):
        super(TestDevice, self).__init__(eid=eid)
        self.test_value = 0.
        self.values = []
        self.times = []
        self.nsets = 0
        self.mi = None
        self.dp = None
        pass

    def get_value(self):
        return self.test_value

    def set_value(self, value):
        self.values.append(value)
        self.nsets += 1
        self.times.append(time.time())
        self.test_value = value


class Target(object):
    def __init__(self,  eid=None):
        """

        :param mi: Machine interface
        :param dp: Device property
        :param eid: ID
        """
        self.eid = eid
        self.id = eid
        self.pen_max = 100

        self.penalties = []
        self.values = []
        self.alarms = []
        self.times = []

    def get_value(self):
        return 0

    def get_penalty(self):
        sase = 0.
        for i in range(self.nreadings):
            sase += self.get_value()
            time.sleep(self.interval)
        sase = sase/self.nreadings
        #sase = self.get_value()
        alarm = self.get_alarm()
        pen = -sase

        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        return pen

    def get_alarm(self):
        return 0

    def clean(self):
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []

class Target_test(Target):
    def __init__(self, mi=None, dp=None, eid=None):
        super(Target_test, self).__init__(eid=eid)
        """
        :param mi: Machine interface
        :param dp: Device property
        :param eid: ID
        """
        self.mi = mi
        self.dp = dp
        self.debug = False
        self.kill = False
        self.pen_max = 100
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []

    def get_penalty(self):
        sase = self.get_value()
        alarm = self.get_alarm()

        if self.debug: print('alarm:', alarm)
        if self.debug: print('sase:', sase)
        pen = 0.0
        if alarm > 1.0:
            return self.pen_max
        if alarm > 0.7:
            return alarm * 50.0
        pen += alarm
        pen -= sase
        if self.debug: print('penalty:', pen)
        self.niter += 1
        print("niter = ", self.niter)
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        return pen

    def get_value(self):
        values = np.array([dev.get_value() for dev in self.devices])
        return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))

    def get_spectrum(self):
        return [0, 0]

    def get_stat_params(self):
        #spetrum = self.get_spectrum()
        #ave = np.mean(spetrum[(2599 - 5 * 120):-1])
        #std = np.std(spetrum[(2599 - 5 * 120):-1])
        ave = self.get_value()
        std = 0.1
        return ave, std

    def get_alarm(self):
        return 0

    def get_energy(self):
        return 3


class SLACTarget(object):
    def __init__(self, mi=None, eid=None):
        super(SLACTarget, self).__init__()
        """
        :param mi: Machine interface
        :param dp: Device property
        :param eid: ID
        """
        self.eid = eid
        self.secs_to_ave = 1
        self.kill = False
        self.pen_max = 100
        self.niter = 0
        self.y = []
        self.x = []
        self.id = eid
        self.pen_max = 100
        self.penalties = []
        self.values = []
        self.alarms = []
        self.times = []
        self.std_dev = []
        self.charge = []
        self.current = []
        self.points = None

    def get_penalty(self):
        sase, std, charge, current = self.get_value()
        alarm = self.get_alarm()
        pen = 0.0
        if alarm > 1.0:
            return self.pen_max
        if alarm > 0.7:
            return alarm * 50.0
        pen += alarm
        pen -= sase
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.std_dev.append(std)
        self.alarms.append(alarm)
        self.charge.append(charge)
        self.current.append(current)
        self.niter += 1
        return pen

    def check_machine(self):
        """ Check the machine for errors """
        self.mi.errorCheck()

    def get_value(self):
        """
        Returns data for the ojective function (detector) from the selected detector PV.
        At lcls the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.
        Args:
                seconds (float): Variable input on how many seconds to average data
        Returns:
                Float of SASE or other detecor measurment
                Float of Standard Deviation of array, or -1 for non-array.
        """
        datain = self.mi.get_value(self.eid)
        dataout, sigma = self.mi.get_sase(datain, self.points)
        charge, current = self.mi.get_charge_current()
        return dataout, sigma, charge, current

    def get_alarm(self):
        """ Used with alarms, not used for now """
        return 0

    def get_energy(self):
        """ Get Energy for hyperparams loading """
        return self.mi.get_energy()
    def get_stat_params(self):
        #get the current mean and std of the chosen detector
        obj_func = mi.get_value(self.eid)
        try:
            std = np.std(  obj_func[(2599-5*120):-1])
            ave = np.mean( obj_func[(2599-5*120):-1])
        except:
            print ("Detector is not a waveform, Using scalar for hyperparameter calc")
            ave = obj_func
            # Hard code in the std when obj func is a scalar
            # Not a great way to deal with this, should probably be fixed
            std = 0.1

        print ("DETECTOR AVE", ave)
        print ("DETECTOR STD", std)

        return ave, std

    def clean(self):
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []

    def get_alarm(self):
        """ Used with alarms, not used for now """
        return 0

    def get_energy(self):
        """ Get Energy for hyperparams loading """
        return self.mi.get_energy()
