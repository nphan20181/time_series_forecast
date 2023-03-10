import module.constants as const
import module.util_functions as utf
from sklearn.linear_model import LinearRegression
import datetime
import numpy as np
import pandas as pd

class Holt:
    
    def __init__(self, alpha=0.5, gamma=0.5, transform=''):
        self.alpha = alpha                          # set smoothing constant for level
        self.gamma = gamma                          # set smoothing constant for growth rate
        self.transform_method = transform           # set transformation method
        self.observe_label = const.Y_LABEL          # set observe label
        self.target = const.TARGET.get(transform)   # set label for target variable
        self.forecast_label = "Holt Forecast"       # set label for in-sample forecast
        
    def fit(self, ts_data: pd.DataFrame):
        '''
        Compute model's components and make in-sample forecasts.
        '''
        
        self.data = ts_data.copy()     # duplicate original data
        self.compute_intial_values()   # compute inital level and growth rate
        self.compute_components()      # compute Levels, Growth Rates and in-sample forecasts

    
    def compute_components(self):
        '''
        Compute model's components and make in-sample forecast.
        '''
        
        level = [self.level_0]                # a list of level (or mean)
        growth_rate = [self.growth_rate_0]    # a list of growth rate
        predictions = []                      # a list of in-sample forecast
        
        # compute level, growth rate and in-sample forecast
        for i, y_t in enumerate(self.data[self.target].tolist()):
            t = i + 1
            
            # compute level at time period t
            level.append(self.alpha * y_t + (1 - self.alpha) * (level[i] + growth_rate[i]))
            
            # compute growth rate at time period t
            growth_rate.append(self.gamma * (level[t] - level[i]) + (1 - self.gamma) * growth_rate[i])
            
            # make in-sample forecast
            predictions.append(level[i] + growth_rate[i])
        
        # save level and growth rate
        self.level = level
        self.growth_rate = growth_rate
        
        # inverse transformation and save forecast value
        self.data[self.forecast_label] = utf.inverse_transform(predictions, self.transform_method)
        
        # compute forecast's error
        self.data[const.ERROR_LABEL] = self.data[self.observe_label] - self.data[self.forecast_label]
        
        # compute standard error
        self.standard_error = np.sqrt(sum((self.data[self.target] - predictions)**2) / (self.data[const.T_LABEL].values[-1] - 2))
        
    def compute_intial_values(self):
        '''
        Fit a regression line to obtain initial value for level and growth rate.
        '''
        
        # prepare data for fitting a linear regression model
        # use the first 104 weeks (or 2 years) of data
        data_subset = self.data[:104]
        X = data_subset[const.T_LABEL].to_numpy().reshape(-1, 1)
        Y = data_subset[self.target]
        
        # fit a linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X, Y)
        
        # save the coefficients
        self.level_0 = lr_model.intercept_
        self.growth_rate_0 = lr_model.coef_[0]
        
    def predict_interval(self, forecast, n):
        '''
        Compute 95% prediction interval of forecast value.
        '''
        
        # compute confidence interval
        ci = const.Z.get('.025') * self.standard_error
        if n >= 2:
            ci = ci * np.sqrt(1 + sum([(self.alpha**2) * (1 + j * self.gamma)**2 for j in range(1, n)]))
        
        # compute upper bound value
        upper = utf.inverse_transform(forecast + ci, self.transform_method) 
        
        # compute lower bound value
        lower = forecast - ci
        lower = utf.inverse_transform(lower, self.transform_method) if lower > 0 else 0
        
        return lower, upper
    
    def predict(self) -> pd.DataFrame:
        '''
        Make out-of-sample forecast for future periods.
        '''
        
        # get the last sales date in the data
        current_date = self.data.Date.iloc[-1]

        predictions = []    # a list of out-of-sample forecasts
        lower = []          # a list of lower limits 
        upper = []          # a list of upper limits
        date_list = []      # a list of future dates
        
        # forecast sale values for the next 52 weeks
        for i in range(0, 52):
            # get future date for prediction
            current_date = current_date + datetime.timedelta(days=7)
            date_list.append(current_date)

            # compute out-of-sample point forecast
            point_forecast = self.level[-1] + (i + 1) * self.growth_rate[-1]
            predictions.append(utf.inverse_transform(point_forecast, self.transform_method))
            
            # compute lower/upper bound of point forecast
            low, high = self.predict_interval(point_forecast, i + 1)
            lower.append(low)
            upper.append(high)
        
        # return forecast data frame
        return pd.DataFrame({'Date': date_list, 'Future Forecast': predictions, 'Lower Bound': lower, 'Upper Bound': upper})

    def evaluate(self) -> pd.DataFrame:
        '''
        Return evaluation metrics.
        '''
        
        # evaluate model using Test data
        mape, mad, rmse = utf.compute_metrics(self.data[self.observe_label], self.data[self.forecast_label])
        
        # create metrics data frame
        df_metrics = pd.DataFrame({'Transform': [const.TRANSFORM_LABEL.get(self.transform_method)],
                                   'Alpha': [self.alpha],
                                   'Gamma': [self.gamma],
                                   'MAPE': [mape], 'MAD': [mad], 'RMSE': [rmse]})
        
        return df_metrics