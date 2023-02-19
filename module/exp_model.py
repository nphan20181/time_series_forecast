import module.util_functions as utf
import numpy as np
import pandas as pd

class ExponentialSmoothing:
    
    def __init__(self, smoothing_constant=0.5, transform='', y_label='Weekly Sales (Million)'):
        self.w = smoothing_constant                # set smoothing constant
        self.observe_label = y_label
        self.target = utf.TARGET.get(transform)    # set target variable
        self.transform_method = transform          # set transformation method
        
        # set model's name
        model_name = "Exponential Smoothing"
        self.model_name = model_name if transform == '' else model_name + '-' + transform
        #self.model_name = self.model_name + '-' + str(smoothing_constant)
        #self.model_label = 'ES' if transform == '' else 'ES-' + transform
        
        # set forecast label
        self.forecast_label = self.model_name + ' Forecast'
    
    def compute_smoothed_series(self):
        '''
        Compute exponentially smoothed series.
        '''
        
        E = []  # a list of exponetially smoothed series
        
        # compute exponentially smoothed series
        for i, y in enumerate(self.train[self.target]):
            smoothed_series = y if i == 0 else self.w * y + (1 - self.w) *E[i-1]
            
            # append smoothed series to a list
            E.append(smoothed_series)
        
        self.exponential_series = E  # save smoothed series
    
    def fit(self, ts_data):
        '''
        Train the model.
        
        Parm:
          - train: a list containing time series data.
        '''
        
        # duplicate original data, then select Train/Test data
        data = ts_data.copy()
        data['Dataset'] = data['Time Series Index'].apply(lambda x: 'Train' if x < 95 else 'Test')
        self.train = data[data.Dataset == 'Train'].copy()  
        
        # transform data
        self.train[self.target] = utf.transform_data(self.train[self.observe_label], self.transform_method)
        
        # compute exponentially smoothed series
        self.compute_smoothed_series()
        
        # compute train error
        self.train[self.forecast_label] = utf.reverse_transform(pd.Series(self.exponential_series), self.transform_method)
        self.train['Train Error'] = self.train[self.observe_label] - self.train[self.forecast_label]
        
        # make predictions on Test data and compute Test error
        test = self.predict(data[data.Dataset == 'Test'])
        test['Test Error'] = test[self.observe_label] - test[self.forecast_label]
        self.test = test  # save test set
     
    def predict(self, ts_data):
        '''
        Make forecast predictions.
        '''
        
        # return the last item in the smoothed series for all future forecast
        prediction = utf.reverse_transform(self.exponential_series[-1], self.transform_method)
        
        # make a copy of forecast data
        forecast = ts_data.copy()
        forecast[self.forecast_label] = [prediction]*ts_data.shape[0]
        
        return forecast
    
    def evaluate(self):
        '''
        Return evaluation metrics on Test data.
        '''
        
        # get evaluation metrics on Test set
        return utf.evaluate_model(self.model_name, self.test[self.forecast_label], self.test[self.observe_label])
    