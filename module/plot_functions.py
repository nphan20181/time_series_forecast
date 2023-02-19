import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

COLORS = {'Observe':'blue', 'Train':'#009933', 'Test':'red', 'Forecast':'#e68a00'}

def create_corr_plot(series, plot_pacf=False, n_lags=52):
    '''
    Create ACF / PACF plot.
    
    Parms:
    - series: a Pandas series of original time series data.
    - plot_pacf: True for PACF plot; False for ACF plot.
    - n_lags: number of lags to be plotted.
    
    Reference: 
        https://community.plotly.com/t/plot-pacf-plot-acf-autocorrelation-plot-and-lag-plot/24108/3
    '''
    corr_array = pacf(series.dropna(), nlags=n_lags, alpha=0.05) if plot_pacf else acf(series.dropna(), 
                                                                                       nlags=n_lags, alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=6)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,n_lags+3], title_text="Time Series Lag")
    fig.update_yaxes(zerolinecolor='#000000', title_text=title)
    fig.update_layout(title=title)
    fig.show()

def plot_forecast(ts_data, train, test, forecast_data, forecast_label, 
                  x_label='Date', y_label='Weekly Sales (Million)', width=800, height=500):
    '''
    Create Observe vs. Forecast plot and return the figure.
    
    Parms:
      - ts_data: original time series data.
      - forecast: model's prediction
    '''
    
    fig = go.Figure()
        
    # create a line plot of observe data
    fig.add_trace(go.Scatter(x=ts_data[x_label], y=ts_data[y_label], mode='lines', name='Observe'))
    
    # create a line plot of Train
    fig.add_trace(go.Scatter(x=train[x_label], y=train[forecast_label], mode='lines', name='Train',
                             line=dict(color=COLORS.get('Train'))))
    
    # create a line plot of Test
    fig.add_trace(go.Scatter(x=test[x_label], y=test[forecast_label], mode='lines', name='Test',
                             line=dict(color=COLORS.get('Test'))))
        
    # create a line plot of forecast data
    fig.add_trace(go.Scatter(x=forecast_data[x_label], y=forecast_data[forecast_label], mode='lines', name=forecast_label,
                             line=dict(color=COLORS.get('Forecast'))))
        
    # update figure's property
    fig.update_xaxes(title_text="<b>" + x_label + "</b>")
    fig.update_yaxes(title_text='<b>Weekly Sales</b>')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                      title_text="<b>Observe vs. Forecast (Million)</b>", width=width, height=height)
        
    return fig

def plot_errors(train_data, test, x_label='Time Series Index', observe_label='Weekly Sales (Million)', width=800, height=500):
    '''
    Create Train vs. Test Error plot and return the figure.
    
    Parms:
      - train_data: train dataset
      - test: test dataset
      - x_label: x-axis label
      - observe_label: label of observe values in time series dataset
      - width: figure's width
      - height: figure's height
    '''
    
    fig = go.Figure()    # create figure
    train = train_data.dropna()
        
    # create line plot of Train/Test Error
    fig.add_trace(go.Scatter(x=train[x_label], y=train['Train Error'], mode='lines', name='Train Error'))
    fig.add_trace(go.Scatter(x=test[x_label], y=test['Test Error'], mode='lines', name='Test Error'))
        
    # update figure's property
    fig.update_xaxes(title_text="<b>" + x_label + "</b>")
    fig.update_yaxes(title_text='<b>' + observe_label + '</b>')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                      title_text="<b>Train vs. Test Error</b>", width=width, height=height)
        
    return fig

def plot_metrics(model_scores, x_label, fig_title, metric='MAPE', height=400, width=600):
    '''
    Create and return a bar plot of evaluation scores for the model.
    
    Parms:
      - model_scores: a data frame containing evaluation scores.
      - x_label: name of a column in the data frame for showing values on x-axis
      - metric: name of the metric for displaying the score
    '''
    
    metrics_df = model_scores[model_scores.Metric == metric]
    
    # create a bar plot
    fig = go.Figure(data=[go.Bar(x=metrics_df[x_label], y=metrics_df['Score'])])
    
    # update axis labels and figure's layout
    fig.update_xaxes(title_text='<b>' + x_label + '<b>')
    fig.update_yaxes(title_text='<b>' + metric + '<b>')
    fig.update_layout(height=height, width=width, title_text=fig_title)
    
    return fig