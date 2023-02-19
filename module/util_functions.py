import numpy as np
import pandas as pd

# name of target variable according to transformation method
TARGET = {'':'Weekly Sales (Million)', 
          'log': 'Log of Weekly Sales (Million)',
          'sqrt': 'Square Root of Weekly Sales (Million)'}

# name of evaluation metrics
METRIC_NAMES = {'MAPE': 'Mean Absolute Percentage Error (MAPE)',
                'MAD': 'Mean Absolute Deviation (MAD)',
                'RMSE': 'Root Mean Square Error (RMSE)'}

# transformation function
TRANSFORM = {'log': np.log, 'sqrt': np.sqrt}
REVERSE_TRANSFORM = {'log': np.exp, 'sqrt': np.square}

def transform_data(ts_data, transform_type=''):
    '''
    Transform time series data.
    
    Parms:
      - ts_data: time series data
      - transform_type: type of transformation, e.g. log (Logarithm) or sqrt (Square Root)
    '''
    
    return ts_data if transform_type == '' else TRANSFORM.get(transform_type)(ts_data)

def reverse_transform(ts_data, transform_type=''):
    '''
    Convert transformed values back to original values.
    '''
    
    return ts_data if transform_type == '' else REVERSE_TRANSFORM.get(transform_type)(ts_data)

def evaluate_model(model_name, forecast, actual):
    '''
    Compute MAPE, MAD, and RMSE and return a data frame containing evaluation metrics.
    
    Parms:
      - model_name: name of the model
      - forecast: predicted values
      - actual: observe values
    '''
    
    n = len(forecast)
    deviations = np.abs(actual - forecast)                 # compute absolute error: actual - predict
    mape = np.round(sum(deviations / actual) / n, 4)       # compute mean absolute percentage error
    mad = np.round(sum(deviations) / n, 4)                 # compute mean absolute deviation
    rmse = np.round(np.sqrt(sum(deviations**2) / n), 4)    # compute root mean square error
    
    # return metrics data frame
    return pd.DataFrame({'Model': [model_name]*3, 
                         'Metric':['MAPE', 'MAD', 'RMSE'], 
                         'Score': [mape, mad, rmse]})