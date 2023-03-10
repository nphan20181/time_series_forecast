import module.constants as const
import module.util_functions as utf
import datetime
import numpy as np
import pandas as pd

class ExponentialSmoothing:
    
    def __init__(self, smoothing_constant=0.5, transform='none'):
        self.alpha = smoothing_constant                 # set smoothing constant
        self.observe_label = const.Y_LABEL              # set observe label
        self.target = const.TARGET.get(transform)       # get label for target variable
        self.transform_method = transform               # set transformation method
        self.forecast_label = 'Exponential Smoothing Forecast'
    
    def compute_smoothed_series(self, initial_level, Y):
        '''
        Compute exponentially smoothed series.
        '''
        
        # create a list of exponetially smoothed series
        E = [initial_level]
        
        # compute exponentially smoothed series and append the result to list E
        for y_t in Y.values:
            E.append(self.alpha * y_t + (1 - self.alpha) * E[-1])
        
        self.exponential_series = E  # save smoothed series
    
    def fit(self, ts_data: pd.DataFrame):
        '''
        Train the model.
        
        Parm:
          - ts_data: a time series data frame.
        '''
        
        # duplicate original data
        data = ts_data.copy()
        
        # compute exponentially smoothed series, use the average sale of week 52 as initial level
        self.compute_smoothed_series(data[data.Week == 52][self.target].mean(), data[self.target])
        
        # compute rolling forecasts
        data[self.forecast_label] = utf.inverse_transform(pd.Series(self.exponential_series[:-1]), self.transform_method)
        data[const.ERROR_LABEL] = data[self.observe_label] - data[self.forecast_label]
        
        # compute standard error
        self.standard_error = np.sqrt(sum(data[const.ERROR_LABEL]**2) / (data.shape[0] - 1))
        
        # save data
        self.data = data
        
    
    def predict_intervals(self, n_periods: int) -> pd.DataFrame:
        '''
        Compute 95% prediction intervals.
        '''
        
        lower = []   # forecast's lower bound
        upper = []   # forecast's upper bound
        
        # get the last value of smoothed series
        level = utf.inverse_transform(self.exponential_series[-1], self.transform_method)    
        conf_interval = const.Z.get('.025') * self.standard_error
        
        # compute confident intervals
        for n in range(1, n_periods + 1):
            if n == 1:
                lower.append(level - conf_interval)
                upper.append(level + conf_interval)
            elif n == 2:
                lower.append(level - conf_interval * np.sqrt(1 + self.alpha**2))
                upper.append(level + conf_interval * np.sqrt(1 + self.alpha**2))
            else:
                lower.append(level - conf_interval * np.sqrt(1 + (n - 1) * self.alpha**2))
                upper.append(level + conf_interval * np.sqrt(1 + (n - 1) * self.alpha**2))
        
        return pd.DataFrame({'Lower Bound': lower, 'Upper Bound': upper})
    
    def predict(self, n_periods=52) -> pd.DataFrame:
        '''
        Make constant forecast for future periods.
        '''
        
        # get the last sales date in the data
        current_date = self.data.Date.iloc[-1]
        
        # get a list of future dates for predictions
        date_list = []
        for i in range(0, n_periods):
            current_date = current_date + datetime.timedelta(days=7)
            date_list.append(current_date)
        
        # create forecast data frame
        df_forecast = pd.concat([pd.DataFrame({'Date': date_list,
                                               'Future Forecast': [self.data[self.forecast_label].values[-1]]*n_periods}), 
                                 self.predict_intervals(n_periods)], axis=1)
        
        return df_forecast
    
    def evaluate(self) -> pd.DataFrame:
        '''
        Return evaluation metrics.
        '''
        
        # evaluate model using Test data
        mape, mad, rmse = utf.compute_metrics(self.data[self.observe_label], self.data[self.forecast_label])
        
        # create metrics data frame
        df_metrics = pd.DataFrame({'Transform': [const.TRANSFORM_LABEL.get(self.transform_method)], 
                                   'Alpha': [self.alpha],
                                   'MAPE': [mape], 'MAD': [mad], 'RMSE': [rmse]})
        
        return df_metrics
    