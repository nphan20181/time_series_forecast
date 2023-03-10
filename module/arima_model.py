import module.constants as const
import module.util_functions as utf
import datetime
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa import stattools

class ARIMA_Model:
    
    def __init__(self, p=0, q=0, transform=''):
        self.ar_order = p                             # set AR order
        self.ma_order = q                             # set MA order
        self.transform_method = transform             # set transform method
        self.target = const.TARGET.get(transform)     # set label for target variable
        self.observe_label = const.Y_LABEL            # set observe label
        self.model_name = 'ARIMA(' + str(p) + ',1,' + str(q) + ')'    # set model's name
        self.forecast_label = self.model_name + ' Forecast'           # set label for in-sample forecast
        
    def fit(self, ts_data):
        '''
        Train ARIMA model and make in-sample forecasts.
        '''
        
        # duplicate original data
        data = ts_data.copy()
        
        # build ARIMA model
        model = ARIMA(data[self.target], exog=data[['Week 47', 'Week 51']], order=(self.ar_order, 1, self.ma_order)).fit()
        
        # make in-sample forecast and compute forecast error
        data[self.forecast_label] = utf.inverse_transform(model.fittedvalues, self.transform_method)   
        data[const.ERROR_LABEL] = data[self.observe_label] - data[self.forecast_label]
        
        # save data
        self.data = data
        self.model = model
        self.residual_info = self.check_residuals()
    
    def check_residuals(self):
        '''
        Check if residuals are independent and uncorrelated.
        '''
        
        # perforem LJung-Box test for normality, p should be > 0.05
        lb_test = acorr_ljungbox(self.data[const.ERROR_LABEL].values, np.arange(1, 53, 1), return_df=True)
        lb_stat = lb_test['lb_stat']
        p_value = lb_test['lb_pvalue']
        
        # check if resdisuals are correlated
        for p_value in lb_test['lb_pvalue'].values:
            if p_value < 0.05:
                return 'Residuals are correlated.'
            
        return 'Residuals are independent and uncorrelated.'
    
    def predict(self, n_periods=52) -> pd.DataFrame:
        '''
        Forecast sales for future periods.
        '''
        
        date_list = []   # a list of future dates for making forecast
        week_list = []   # a list of week numbers
        week_47 = []     # a list of 0s or 1s to indicate whether week number is 47 or not
        week_51 = []     # a list of 0s or 1s to indicate whether week number is 51 or not

        # get the last sales date in the data
        current_date = self.data.Date.iloc[-1]
        for i in range(0, n_periods):
            # get the date of next week
            current_date = current_date + datetime.timedelta(days=7)   
            date_list.append(current_date)
            
            # get week number
            week = current_date.isocalendar().week                     
            week_list.append(week)
            
            # set value for dummy variables
            week_47.append(1 if week == 47 else 0)
            week_51.append(1 if week == 51 else 0) 
        
        # make out-of-sample forecast
        forecast = self.model.get_prediction(start=147, end=147+51, exog=pd.DataFrame({'Week 47': week_47, 'Week 51': week_51})) 
        yhat = utf.inverse_transform(forecast.predicted_mean, self.transform_method)
        
        # get confidence intervals
        yhat_conf_int = forecast.conf_int(alpha=0.05)
        lower = utf.inverse_transform(yhat_conf_int['lower ' + self.target].values, self.transform_method)
        upper = utf.inverse_transform(yhat_conf_int['upper ' + self.target].values, self.transform_method)

        # create and return forecast data frame
        return pd.DataFrame({'Date': date_list, 'Week': week_list,
                             'Future Forecast': yhat, 'Lower Bound': lower, 'Upper Bound': upper})
    
    def evaluate(self) -> pd.DataFrame:
        '''
        Return evaluation metrics.
        '''
        
        # evaluate model using Test data
        mape, mad, rmse = utf.compute_metrics(self.data[self.observe_label], self.data[self.forecast_label])
        
        # create and return metrics data frame
        return pd.DataFrame({'Transform': [const.TRANSFORM_LABEL.get(self.transform_method)], 
                             'p': [self.ar_order], 'q': [self.ma_order],
                             'AIC': [np.round(self.model.aic, 2)], 'MAPE': [mape], 'MAD': [mad], 'RMSE': [rmse]})