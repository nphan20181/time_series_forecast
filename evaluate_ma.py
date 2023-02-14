import pandas as pd
import numpy as np
from tqdm import tqdm
from module.ma_model import MovingAverage

def create_ma_metrics_file(train, test):
    '''
    Evaluate MA models, create a score metrics file and export results to a csv file.
    '''
    
    metrics = None   # a data frame for storing metric scores

    # evaluate model for 52 different moving points
    for m in tqdm(range(1, 53)):
        # evaluate model for different moving types: average/median
        for moving_type in MovingAverage.MTYPE.keys():
            # evaluate model for different type of data transformation: none/logarithm/square root
            for transform_method in MovingAverage.TARGET.keys():
                model = MovingAverage(m, transform=transform_method, moving_type=moving_type)
                model.fit(train, test)

                # build evaluation metrics data frame
                if metrics is None:  # create a new data frame to store model's metric scores
                    metrics = model.evaluate()
                else:  # append evaluation metrics to existing data frame, drop duplicates
                    metrics = pd.concat([metrics, model.evaluate()], axis=0)
                    metrics.drop_duplicates(inplace=True, ignore_index=True, keep='first', subset='Model')
    
    # save evaluation results to csv
    metrics.to_csv('data/ma_metrics.csv', index=False)


# begin main
if __name__ == "__main__":
    
    # load Train/Test data
    train = pd.read_pickle('data/train.pkl')
    test = pd.read_pickle('data/test.pkl')

    # build evaluation metrics file for MA model
    create_ma_metrics_file(train, test)