import module.constants as const
import module.util_functions as utf
from sklearn.linear_model import LinearRegression
import datetime
import numpy as np
import pandas as pd

class HoltWinters:
    
    def __init__(self, alpha=0.5, gamma=0.5, delta=0.5, transform=''):
        self.alpha = alpha                             # set smoothing constant for level
        self.gamma = gamma                             # set smoothing constant for growth rate
        self.delta = delta                             # set smoothing constant for seasonal component
        self.transform_method = transform              # set transformation method
        self.observe_label = const.Y_LABEL             # set observe label
        self.target = const.TARGET.get(transform)      # set label for target variable
        self.forecast_label = 'Holt Winters Forecast'  # set label fore in-sample forecast
        self.P = 52                                    # set time periods
        
    def fit(self, ts_data: pd.DataFrame):
        '''
        Compute model's components and make in-sample forecasts.
        '''
        
        self.data = ts_data.copy()        # duplicate original data
        self.compute_seasonal_factor()    # compute seasonal factor
        self.compute_components()         # compute/update components
        
        # compute in-sample forecast's error
        self.data[const.ERROR_LABEL] = self.data[self.observe_label] - self.data[self.forecast_label]

    
    def compute_components(self):
        '''
        Compute and model's components.
        '''
        
        level = [self.level_0]               # a list of levels
        growth_rate = [self.growth_rate_0]   # a list of growth rates
        predictions = []                     # a list of in-sample forecasts
        sre = []                             # a list of squared relative error
        
        for i, y_t in enumerate(self.data[self.target].tolist()):
            t = i + 1    # set current time period
            
            # get seasonal index at time period t - 52
            seasonal_index = self.seasonal_factor[self.seasonal_factor['t'] == t - self.P]['Detrended'].tolist()[0]
            
            # compute level at time period t
            level.append(self.alpha * (y_t / seasonal_index) + (1 - self.alpha) * (level[i] + growth_rate[i]))
            
            # compute growth rate at time period t
            growth_rate.append(self.gamma * (level[t] - level[t-1]) + (1 - self.gamma) * growth_rate[i])
            
            # compute seasonal factor at time period t
            sn_t = self.delta * (y_t / level[t]) + (1 - self.delta) * seasonal_index
            self.seasonal_factor = pd.concat([self.seasonal_factor, pd.DataFrame({'t': [t], 'Detrended': [sn_t]})], axis=0)
            
            # make in-sample prediction
            forecast = utf.inverse_transform(((level[i] + growth_rate[i]) * seasonal_index), self.transform_method)
            predictions.append(forecast)
            
            # compute squared relative error
            error = ((y_t - (level[i] + growth_rate[i]) * seasonal_index) / ((level[i] + growth_rate[i]) * seasonal_index))**2
            sre.append(error)
        
        # save level, growth rate, predictions, and squared relative error
        self.level = level
        self.growth_rate = growth_rate
        self.data[self.forecast_label] = predictions
        self.data['Squared Relative Error'] = sre
        
        # compute standard error
        self.standard_error = np.sqrt(sum(sre) / (self.data[const.T_LABEL].values[-1] - 3))
        
    def compute_seasonal_factor(self):
        '''
        Fit a regression line to obtain initial value for level and growth rate and compute initial seasonal factors.
        '''
        
        # prepare data for fitting a linear regression model
        X = self.data[const.T_LABEL].to_numpy().reshape(-1, 1)
        Y = self.data[self.target]
        
        # fit a linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X, Y)
        
        # compute initial seasonal factors
        self.data['Regression Estimates'] = lr_model.predict(X)
        self.data['Detrended'] = self.data[self.target] / self.data['Regression Estimates']    # detrend the data
        seasonal_factor = self.data.groupby(['Week'])['Detrended'].mean().reset_index()        # compute average Detrend per week
        seasonal_factor['t'] = seasonal_factor['Week'] - 52                                    # set time period to the past
        del seasonal_factor['Week']
        
        # save seasonal factors and initial value for level/growth rate
        self.seasonal_factor = seasonal_factor
        self.level_0 = lr_model.intercept_
        self.growth_rate_0 = lr_model.coef_[0]
        
    def predict_intervals(self, forecast, n, seasonal_factor):
        '''
        Compute 95% prediction intervals.
        '''
        
        # get the last value of level and growth rate
        level = self.level[-1]
        growth_rate = self.growth_rate[-1]
        
        # compute confidence interval
        if n == 1:
            c = (level + growth_rate)**2
        elif n >= 2 and n <= self.P:
            c = sum([((self.alpha**2) * (1 + (n-j) * self.gamma )**2) * ((level + j * growth_rate)**2) + \
                 (level + n * growth_rate)**2 for j in range(1, n)])
        else:
            c = 1
        
        # compute confidence interval
        ci = const.Z.get('.025') * self.standard_error * np.sqrt(c) * seasonal_factor
        
        # compute lower limit
        lower = forecast - ci
        lower = utf.inverse_transform(lower, self.transform_method) if lower > 0 else 0
        
        # compute upper limit
        upper = utf.inverse_transform(forecast + ci, self.transform_method)
        
        return lower, upper
    
    def predict(self):
        '''
        Make out-of-sample forecast for future periods.
        '''
        
        predictions = []    # a list of out-of-sample forecasts
        lower = []          # a list of lower limits 
        upper = []          # a list of upper limits
        date_list = []      # a list of future dates
        
        # get the last sales date in the data
        current_date = self.data.Date.iloc[-1]

        # get the last time period
        t = self.data[const.T_LABEL].iloc[-1]
        
        for i in range(0, self.P):
            # get future date for prediction
            current_date = current_date + datetime.timedelta(days=7)
            date_list.append(current_date)

            # get seasonal index at time period t - 52
            t = t + 1
            seasonal_index = self.seasonal_factor[self.seasonal_factor['t'] == t - self.P]['Detrended'].tolist()[0]
            
            # compute point forecast
            point_forecast = (self.level[-1] + (i + 1) * self.growth_rate[-1]) * seasonal_index
            predictions.append(utf.inverse_transform(point_forecast, self.transform_method))
            
            # get prediction interval
            low, high = self.predict_intervals(point_forecast, i + 1, seasonal_index)
            lower.append(low)
            upper.append(high)
        
        # create and return forecast data frame
        return pd.DataFrame({'Date': date_list, 'Future Forecast': predictions, 'Lower Bound': lower, 'Upper Bound': upper})

    def evaluate(self):
        '''
        Return evaluation metrics.
        '''
        
        # evaluate model using Test data
        mape, mad, rmse = utf.compute_metrics(self.data[self.observe_label], self.data[self.forecast_label])
        
        # create and return metrics data frame
        return pd.DataFrame({'Transform': [const.TRANSFORM_LABEL.get(self.transform_method)],
                             'Alpha': [self.alpha],
                             'Gamma': [self.gamma],
                             'Delta': [self.delta],
                             'MAPE': [mape], 'MAD': [mad], 'RMSE': [rmse]})