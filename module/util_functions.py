import numpy as np
import pandas as pd

def evaluate_model(model_name, forecast, actual):
    '''
    Compute MAPE, MAD, and RMSE and return a data frame containing evaluation metrics.
    
    Parms:
      - model_name: name of the model
      - forecast: predicted values
      - actual: observe values
    '''
    
    n = len(forecast)
    deviations = np.abs((actual - forecast))               # compute absolute error: actual - predict
    mape = np.round(sum(deviations / actual) / n, 4)       # compute mean absolute percentage error
    mad = np.round(sum(deviations) / n, 4)                 # compute mean absolute deviation
    rmse = np.round(np.sqrt(sum(deviations**2) / n), 4)    # compute root mean square error
    
    # return metrics data frame
    return pd.DataFrame({'Model': [model_name]*3, 
                         'Metric':['MAPE', 'MAD', 'RMSE'], 
                         'Score': [mape, mad, rmse]})