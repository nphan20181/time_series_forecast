import pandas as pd
import numpy as np
import module.util_functions as utf
from tqdm import tqdm
from module.ma_model import MovingAverage
from module.exp_model import ExponentialSmoothing

def build_metrics_table(model, metrics, new_col):
    '''
    Build and return a data frame containing model's evaluation metrics.
    
    Parms:
      - model: a model object
      - metrics: an existing metrics data frame
      - new_col: a tuple of column's label and corresponding value
    '''
    
    model_evaluate = model.evaluate()         # create a data frame containing evaluation metrics
    model_evaluate[new_col[0]] = new_col[1]   # add specified column to data frame
    
    # build evaluation metrics data frame
    if metrics is None:  # create a new data frame to store model's metric scores
        metrics = model_evaluate
    else:  # append evaluation metrics to existing data frame, drop duplicates
        metrics = pd.concat([metrics, model_evaluate], axis=0)
        metrics.drop_duplicates(inplace=True, ignore_index=True)
    
    return metrics


def create_ma_metrics_file(ts_data):
    '''
    Evaluate MA models, create a score metrics file and export results to a csv file.
    '''
    
    metrics = None   # a data frame for storing metric scores

    # evaluate model for 52 different moving points
    for m in tqdm(range(1, 53)):
        # evaluate model for different moving types: average/median
        for moving_type in MovingAverage.MTYPE.keys():
            # evaluate model for different type of data transformation: none/logarithm/square root
            for transform_method in utf.TARGET.keys():
                model = MovingAverage(m, transform=transform_method, moving_type=moving_type)
                model.fit(ts_data)
                metrics = build_metrics_table(model, metrics, ('Moving Window (m)', m))
    
    def getModelType(s):
        '''
        Get model's type from model's name string.
        '''
        
        s_split = s.split('-')   # split the string on hyphen
        model_type = s_split[1] if len(s_split) == 2 else s_split[1] + '-' + s_split[2]  # get model's type
        return model_type
    
    # get moving window (m) and model's type from model's name
    #metrics['Moving Window (m)'] = metrics['Model'].apply(lambda x: x.split('-')[0])
    metrics['Type'] = metrics['Model'].apply(getModelType)

    # save evaluation results to csv
    metrics.to_csv('data/ma_metrics.csv', index=False)

def create_exp_metrics_file(ts_data):
    '''
    Evaluate Exponential Smoothing models, create a score metrics file and export results to a csv file.
    '''
    
    metrics = None   # a data frame for storing metric scores

    # evaluate model for different smoothing constants
    for smoothing_constant in tqdm(np.arange(0.0, 1.01, 0.01)):
        smoothing_constant = np.round(smoothing_constant, 2)
        # evaluate model for different type of data transformation: none/logarithm/square root
        for transform_method in utf.TARGET.keys():
            model = ExponentialSmoothing(smoothing_constant, transform_method)
            model.fit(ts_data)
            metrics = build_metrics_table(model, metrics, ('Smoothing Constant (w)', smoothing_constant))
    
    # save evaluation results to csv
    metrics.to_csv('data/exp_metrics.csv', index=False)
            

# begin main
if __name__ == "__main__":
    
    # load time series data
    ts_data = pd.read_csv('data/ts_dataset.csv')
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])

    print('Build MA metrics table...')
    create_ma_metrics_file(ts_data)       # build evaluation metrics file for MA model
    
    print('Build Exponential Smoothing metrics table...')
    create_exp_metrics_file(ts_data)      # build evaluation metrics file for Exponential Smoothing model