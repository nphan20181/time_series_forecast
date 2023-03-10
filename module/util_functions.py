from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from module.constants import TARGET, METRIC_NAMES, TRANSFORM, REVERSE_TRANSFORM, ERROR_LABEL
import numpy as np
import pandas as pd
import os

def load_ts_data():
    '''
    Load time series data and change data type of "Date" from string to date.
    
    '''
    
    file_path = os.path.join('data', 'ts_dataset.csv')
    ts_data = pd.read_csv(file_path)                    # load time series data
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])   # change data type of Date from string to date.
    
    return ts_data

def transform_data(ts_data, transform_type=''):
    '''
    Transform time series data.
    
    Parms:
      - ts_data: time series data
      - transform_type: type of transformation, e.g. log (Logarithm) or sqrt (Square Root)
    '''
    
    return ts_data if transform_type == '' else TRANSFORM.get(transform_type)(ts_data)

def inverse_transform(ts_data, transform_type=''):
    '''
    Convert transformed values back to original values.
    '''
    
    return ts_data if transform_type == '' else REVERSE_TRANSFORM.get(transform_type)(ts_data)

def adf_test(df_series):
    '''
    Perform Augmented Dickey–Fuller test to check whether data is stationary or not.
    '''
    
    ADF_result = adfuller(df_series, maxlag=52)
    
    # return ADF Statistic and p-value
    return np.round(ADF_result[0], 2), np.round(ADF_result[1], 8)

def check_residuals(data):
    '''
    Check if residuals are independent and uncorrelated.
    '''
        
    # perforem LJung-Box test for normality, p should be > 0.05
    lb_test = acorr_ljungbox(data[ERROR_LABEL].values, np.arange(1, 53, 1), return_df=True)
    lb_stat = lb_test['lb_stat']
    p_value = lb_test['lb_pvalue']
        
    # check if resdisuals are correlated
    for p_value in lb_test['lb_pvalue'].values:
        if p_value < 0.05:
            return 'Residuals are correlated.'
            
    return 'Residuals are independent and uncorrelated.'

def compute_metrics(actual, predict):
    '''
    Compute MAPE, MAD, and RMSE and return the scores.
    
    Parms:
      - actual: observe values
      - predict: predicted values
    '''
    
    n = len(actual)
    deviations = np.abs(actual - predict)                       # compute absolute error: actual - predict
    mape = np.round(np.mean((deviations / actual)) * 100, 2)    # compute mean absolute percentage error
    mad = np.round(sum(deviations) / n, 4)                      # compute mean absolute deviation
    rmse = np.round(np.sqrt(sum(deviations**2) / n), 4)         # compute root mean square error
    
    return mape, mad, rmse