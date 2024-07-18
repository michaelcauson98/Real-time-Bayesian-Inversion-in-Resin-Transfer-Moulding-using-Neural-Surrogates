# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:32:10 2023

@author: pmymc12
"""

import numpy as np

class Data:
    
    """
    Description
    -----------
    Data class holds the all data from the Matlab simulations in raw form, but
    also in parameterised form (which is better-suited to torch application).
    The parameterised form is split into train, validation and test sets. The
    data class can also generate synthetic data for virtual experiments by
    adding noise to it.

    Parameters
    ----------
    X : numpy.ndarray
        Input to Matlab simulations. For each row of X, the first MxM + 2M_RT
        elements are permeability values in each cell, the second MxM + 2M_RT
        elements are porosity values in each cell, and the remaining 2 are 
        the inlet pressure and viscosity provided.
    Y : numpy.ndarray
        Output of Matlab simulations. Each row of Y corresponds to 116 pressure
        sensors measuring at 14 observation times (i.e. length 1624).
    filling_times : numpy.ndarray
        Matlab simulations also include the filling times of each experiment.
    train_split: float
        Proportion of the data allocated to training.
    Experiment : class
        This is the class object in Experiment.py.
    param_min, param_max : numpy.ndarray
        These are the min/max of the columns of X, used to parameterise the 
        data into format better-suited to torch implementation
    TrainX, TrainY : numpy.ndarray
        Training data set, used for training neural network.
    DevelopX, DevelopY : numpy.ndarray
        Development data set, used to independently check neural network quality.
    TestX, TestY : numpy.ndarray
        Testing data set, used to check quality of the inversion.
    
    Functions
    -------
    Public UnitTransformX :
        Affine transformation of X into the unit hypercube.
    Public UnitTransformY :
        Log-transform of Y "squashes" data conveniently and ensures positivity.
    Public generate_data :
        Adds observational noise to true pressure values according to sigma1
        and sigma2.

    Returns
    -------
    No returns

    Examples
    --------
    >>> Data_object = Data(Experiment_object,X_data,Y_data,filling_times_data)
    """
    
    def __init__(self, Experiment, X, Y, filling_times, train_split = 0.7):
        
        self.X = X
        self.Y = Y
        self.filling_times = filling_times
        self.train_split = train_split
        self.Experiment = Experiment
        self.param_min = Experiment.param_min
        self.param_max = Experiment.param_max

        # Training parameterisation
        self._logY_means = np.mean(np.log(self.Y+1_000),axis=0)
        self._logY_stds = np.std(np.log(self.Y+1_000))
        self._logX_means = np.mean(np.log( (self.param_max-self.X)/(self.X-self.param_min) ),axis=0)
        self._logX_stds = np.std(np.log( (self.param_max-self.X)/(self.X-self.param_min) ),axis=0)
        
        self._paramY = self.ParameteriseY(self.Y,"FWD")
        self._unitX = self.UnitTransformX(self.X, "FWD")
        
        # Data split
        validation = 0.98 # 1,000 (2% of 50,000) allocated to testing
        self.TrainX = self._unitX[0:round(self.train_split*len(self._unitX))]
        self.DevelopX = self._unitX[round(self.train_split*len(self._unitX)):
                                    round(validation*len(self._unitX))]
        self.TestX = self._unitX[round(validation*len(self._unitX)):]
        
        self.TrainY = self._paramY[0:round(self.train_split*len(self._paramY))]
        self.DevelopY = self._paramY[round(self.train_split*len(self._paramY)):
                                     round(validation*len(self._paramY))]
        self.TestY = self._paramY[round(validation*len(self._paramY)):]
        
        print("Train, validation and test sets created.")
            
        
    # X -> [0,1]^N
    def UnitTransformX(self,data,direction):
        
        if direction == "FWD":
            return (data - self.param_min)/(self.param_max-self.param_min)
        else:
            return data*(self.param_max-self.param_min) + self.param_min

        
    # Y -> log(Y) -> normalise log(Y)
    def ParameteriseY(self,data,direction):
        
        if direction == "FWD":
            logdata = np.log(data+1_000)
            return (logdata-self._logY_means)/self._logY_stds
        else:
            logdata = data*self._logY_stds + self._logY_means
            return np.exp(logdata)-1_000
    
    # Generates pressure data (at time t) at selected sensors, noise is scaled by sigma1 and sigma2.
    def generate_data(self,pressure_data,sensor_inds,sigma1,sigma2,surr_error = None):
        
        # Collect all sensor inds
        all_sensor_inds = []
        for t in range(1,self.Experiment.times+1):
            all_sensor_inds.append(np.array(sensor_inds) + self.Experiment.sensors_per_time*(t-1))
        all_sensor_inds = np.concatenate(all_sensor_inds)
        
        # Perturb data
        if surr_error is None:
            truth = pressure_data[all_sensor_inds]
        else:
            truth = pressure_data[all_sensor_inds] - surr_error[all_sensor_inds]
            
        test_y_transformed = truth - np.min(truth)
        Gamma1 = (sigma1*np.abs(test_y_transformed))**2
        Gamma2 = (sigma2*np.abs(np.max(test_y_transformed)-np.min(test_y_transformed)))**2
        Gamma = Gamma1 + Gamma2
        noise = np.random.normal(0,1,len(all_sensor_inds))*np.sqrt(Gamma1) + np.random.normal(0,1,len(all_sensor_inds))*np.sqrt(Gamma2)
        sim_data = truth + noise
        
        return truth, sim_data, Gamma, all_sensor_inds
     