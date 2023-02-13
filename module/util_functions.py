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
    mape = np.round(sum(np.abs((actual - forecast) / actual)) / n, 4)
    mad = np.round(sum(np.abs(actual - forecast)) / n, 4)
    rmse = np.round(np.sqrt(sum((actual - forecast)**2) / n), 4)
    
    # create data frame
    return pd.DataFrame(dict({'Model': [model_name], 'MAPE': [mape], 'MAD': [mad], 'RMSE': [rmse]}))