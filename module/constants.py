import numpy as np

# set labels
X_LABEL ='Date'
Y_LABEL ='Weekly Sales (Million)'
T_LABEL = 'Time Period'
ERROR_LABEL = 'Residual'

# name of target variable according to transformation method
TARGET = {'':'Weekly Sales (Million)', 
          'Log': 'Log of Weekly Sales (Million)',
          'Sqrt': 'Square Root of Weekly Sales (Million)'}

# name of evaluation metrics
METRIC_NAMES = {'MAPE': 'Mean Absolute Percentage Error (MAPE)',
                'MAD': 'Mean Absolute Deviation (MAD)',
                'RMSE': 'Root Mean Square Error (RMSE)'}

# transformation function
TRANSFORM = {'Log': np.log, 'Sqrt': np.sqrt}
REVERSE_TRANSFORM = {'Log': np.exp, 'Sqrt': np.square}
TRANSFORM_LABEL = {'': 'None', 'Log': 'Logarithm', 'Sqrt': 'Square Root'}

NUMBER_ORDER = {0: 'Zeroth', 1: 'First', 2: 'Second', 3: 'Third', 4: 'Fourth', 5: 'Fifth', 
                6: 'Sixth', 7: 'Seventh', 8: 'Eighth', 9: 'Ninth', 10: 'Tenth'}

COLORS = {'Observe':'blue', 'Train':'#009933', 'Test':'red', 'Forecast':'#e68a00'}

# z-value for computing 95% confidence interval
Z = {'.025': 1.96}