import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress
import module.util_functions as utf


class MovingAverage:
    # moving type
    MTYPE = {'MA': 'Moving Average', 'MM': 'Moving Median'}
    
    def __init__(self, m, transform='', moving_type='MA'):
        '''
        Parms:
          - m: order of moving average
          - transform: type of data transformation
          - moving_type: average (MA) or median (MM)
        '''
        
        self.m = m
        self.y_label = utf.TARGET.get(transform)
        self.observe_label = utf.TARGET.get('')
        self.moving_type = moving_type
        self.transform = transform
        
        # set ma_label
        self.ma_label = str(m) + '-' + moving_type
        self.ma_label = self.ma_label if self.transform == '' else self.ma_label + '-' + self.transform
        
        self.ma_extend_label = self.ma_label + ' Extended Trend'     # label for extended trend values
        self.ma_forecast_label = self.ma_label + ' Forecast'         # label for forecast values
        
        # set model's name
        m_name = str(m) + '-' + self.MTYPE.get(moving_type)
        self.model_name = m_name if self.transform == '' else m_name + '-' + self.transform
    
    def compute_moving_values(self):
        '''
        Compute moving values based on previous m windows.
        '''
        
        # duplicate train data
        self.ma = self.train[['Time Series Index', 'Week', 'Date', self.y_label]].copy()
        
        # compute moving values for m windows
        if self.moving_type == 'MM':   # Moving Median
            self.ma[self.ma_label] = self.train[self.y_label].rolling(self.m).median()
        else:    # Moving Average
            self.ma[self.ma_label] = self.train[self.y_label].rolling(self.m).mean()
            
        # copy moving values to the training set
        self.train[self.ma_label] = self.ma[self.ma_label]
            
        # compute ratio y_t / M_t (current y value / current moving value)
        self.ma['Ratio'] = self.train[self.y_label] / self.ma[self.ma_label]
        self.ma.dropna(inplace=True)
    
    def get_extend_trend_line(self):
        '''
        Compute slope and intercept of the extended trend line using the last 2 points of the moving values.
        Return slope and intercept of the extended trend line.
        
        Parms:
          - X: a list of two x values (x-coordinate)
          - Y: a list of two y values (y-coordinate)
        '''
        
        # get equation of the extended trend line (y=mx+b) using the last 2 points of the moving line
        slope, intercept, _, _, _ = linregress([0, 1], self.ma[self.ma_label][-2:].to_list())
        
        # set trend line's equation
        self.ext_trend_line = "y = " + str(np.round(intercept, 4)) + " + (" + str(np.round(slope, 4)) + ")x"
        
        return (slope, intercept)
    
    def compute_ma_extend(self, n=52):
        '''
        Compute y-values of the extended trend line.
        
        Parm:
          - n: number of weeks
        '''
        
        # get slope and intercept of the extended trend line
        slope, intercept = self.get_extend_trend_line()
        
        # compute y-values of the extended trend line for the next n weeks
        self.ma_extend = [intercept + slope * x for x in range(1, n)]
        
    def predict(self, ts_data, save_data=False):
        '''
        Forecast future values.
        
        Parm:
          - ts_data: a data frame containing information of the weeks for making forecast
        '''
        
        # copy the data for making prediction
        forecast = ts_data.copy()
        
        # get seasonal index
        forecast['Seasonal Index'] = forecast['Week'].map(self.seasonal_index['Ratio'])
        
        # get values of the extended trend for the next n weeks
        self.compute_ma_extend(len(forecast) + 2)
        forecast[self.ma_extend_label] = self.ma_extend[:forecast.shape[0]]
        
        # predict future value by multiplying extended trend value and corresponding seasonal index
        forecast[self.ma_forecast_label] = forecast[self.ma_extend_label] * forecast['Seasonal Index']
        
        # convert transformed value back to original dollar value
        forecast[self.ma_forecast_label] = utf.reverse_transform(forecast[self.ma_forecast_label], self.transform)
        
        if(save_data):
            self.forecast = forecast
        
        return forecast
    
    def fit(self, ts_data):
        '''
        Train model and forecast sale values.
        
        Parms:
          - train: data used for training the model.
          - test: data used for evaluating the model.
        '''
        
        # duplicate original data, then select Train/Test data
        self.data = ts_data.copy()
        self.data['Dataset'] = self.data['Year'].apply(lambda year: 'Train' if year < 2012 else 'Test')
        self.train = self.data[self.data.Dataset == 'Train'].copy()
        self.test = self.data[self.data.Dataset == 'Test'].copy()
        
        # compute moving values
        self.compute_moving_values() 
        
        # compute seasonal index
        if self.moving_type == 'MM':   # seasonal index for Moving Median
            seasonal_index = self.ma.groupby(['Week'])['Ratio'].median()
        else:  # seasonal index for Moving Average
            seasonal_index = self.ma.groupby(['Week'])['Ratio'].mean()
        
        # save seasonal index for later retrieval
        self.seasonal_index = pd.DataFrame(seasonal_index).to_dict()    
        
        # get seasonal index for train data
        self.train['Seasonal Index'] = self.train['Week'].map(self.seasonal_index['Ratio'])
        
        # make predictions on Train dataset
        # for Train data, forecast = moving value * seasonal index
        self.train[self.ma_forecast_label] = utf.reverse_transform(self.train[self.ma_label] * self.train['Seasonal Index'],
                                                                   self.transform)
        self.train['Train Error'] = self.train[self.observe_label] - self.train[self.ma_forecast_label]    # compute Train Error
        
        # predict future values on Test dataset
        self.test = self.predict(self.test)
        self.test['Test Error'] = self.test[self.observe_label] - self.test[self.ma_forecast_label]        # compute Test Error
        
    def evaluate(self):
        '''
        Return evaluation metrics on Test data.
        '''
        
        # get evaluation metrics on Test set
        return utf.evaluate_model(self.ma_label, self.test[self.ma_forecast_label], self.test[self.observe_label])
    
    def plot_ma_components(self, x_label='Time Series Index', width=650, height=440):
        '''
        Create a plot of Time Series Components of the model and return the figure.
        '''
        
        fig = go.Figure()
        
        # plot original time series
        fig.add_trace(go.Scatter(x=self.data[x_label], y=self.data[self.y_label],
                         mode='lines', name='Observe'))
        
        # plot moving line
        fig.add_trace(go.Scatter(x=self.ma[x_label], y=self.ma[self.ma_label], mode='lines', name=self.ma_label))
        
        # plot moving extended trend line
        fig.add_trace(go.Scatter(x=self.test[x_label]-1, y=self.test[self.ma_extend_label],
                                 mode='lines', name=self.ma_extend_label))
        
        # update figure's property
        fig.update_xaxes(title_text='<b>' + x_label + '</b>')
        fig.update_yaxes(title_text='<b>' + self.y_label + '</b>')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                          title_text="<b>Time Series Components</b>", width=width, height=height)
        
        return fig
    
    def plot_seasonal_indexes(self, x_label='Time Series Index', width=650, height=440):
        '''
        Create a plot of n-Points Seasonal Index and return the figure.
        '''
        
        fig = go.Figure()
        
        # get seasonal index and then draw a line plot
        seasonal_index_df = pd.DataFrame(self.seasonal_index)
        fig.add_trace(go.Scatter(x=seasonal_index_df.index, y=seasonal_index_df['Ratio'], mode='lines'))
        
        # update layout
        fig.update_xaxes(title_text='<b>' + x_label + '</b>')
        fig.update_yaxes(title_text='<b>Seasonal Index</b>')
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                          title_text="<b>" + str(self.m) + " Points Seasonal Index</b>", width=width, height=height)
        
        return fig