from sklearn.linear_model import LinearRegression
import module.constants as const
import module.util_functions as utf
import datetime
import numpy as np
import pandas as pd

class MultiplicativeDecomposition:
    
    def __init__(self, transform=''):
        self.target = const.TARGET.get(transform)       # set label for target variable
        self.transform_method = transform               # set transformation method
        
    def fit(self, ts_data: pd.DataFrame):
        '''
        Decompose time series data into trend, seasonal, cyclical, and irregular components.
        '''

        # duplicate original data
        self.data = ts_data.copy()
        
        # decompose time series data into seasonal, trend, cyclical, and irregular components
        self.compute_seasonal_factors()    
        self.compute_trend()              
        self.compute_cyclical_and_irregular()           
    
    def compute_seasonal_factors(self):
        '''
        Compute seasonal factors.
        '''
        
        # compute centered moving average
        cma = pd.DataFrame(self.data[self.target].rolling(window=52, center=True).mean().dropna().rolling(window=2).mean().dropna())
        cma.rename(columns={self.target : 'CMA'}, inplace=True)
        cma.index = cma.index - 1                                 # move the center up 1 row
        cma['Week'] = self.data['Week'].iloc[cma.index]           # get week number for corresponding time period
        
        # compute seasonal x irregular
        cma['sn x ir'] = self.data[self.target].iloc[cma.index] / cma['CMA']
        
        # compute seasonal factors for each week in a year
        seasonal_factors = cma.groupby(['Week'])['sn x ir'].mean()
        self.data['Seasonal Factor'] = self.data['Week'].map(seasonal_factors)
    
    def compute_trend(self):
        '''
        Compute trend component.
        '''
        
        # compute detrended component
        detrend = self.data[self.target] / self.data['Seasonal Factor']
        
        # prepare data for fitting a linear regression model
        X = self.data[const.T_LABEL].to_numpy().reshape(-1, 1)
        
        # fit a linear regression model on Detrended component
        lr_model = LinearRegression()
        lr_model.fit(X, detrend)
        
        # get regression estimate for Trend component
        self.data['Trend'] = lr_model.predict(X)
        
    def compute_cyclical_and_irregular(self):
        '''
        Compute cyclical and irregular components.
        '''
        
        # compute cyclical x irregular
        cl_and_ir = self.data[self.target] / (self.data['Trend'] * self.data['Seasonal Factor'])

        # compute cyclical component
        cyclical = cl_and_ir.rolling(window=3).mean().dropna()
        cyclical.index = cyclical.index - 1                     # move the center for cyclical component up 1 row
        
        # compute irregular component
        irregular = cl_and_ir.iloc[cyclical.index] / cyclical
        
        # save cyclical and irregular components
        self.cyclical_component = pd.DataFrame({const.T_LABEL: self.data[const.T_LABEL].iloc[cyclical.index], 'Cyclical': cyclical})
        self.irregular_component = pd.DataFrame({const.T_LABEL: self.data[const.T_LABEL].iloc[cyclical.index], 'Irregular': irregular})