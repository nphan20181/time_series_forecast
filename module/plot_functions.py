import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

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

def plot_metrics(model_scores, x_label, metric='MAPE', height=400, width=600):
    '''
    Create and return a bar plot of evaluation scores for the model.
    
    Parms:
      - model_scores: a data frame containing evaluation scores.
      - x_label: name of a column in the data frame for showing values on x-axis
      - metric: name of the metric for displaying the score
    '''
    
    # create a bar plot
    fig = px.bar(model_scores[model_scores.Metric == metric], x=x_label, y='Score')
    
    # update axis labels and figure's layout
    fig.update_xaxes(title_text='<b>' + x_label + '<b>')
    fig.update_yaxes(title_text='<b>MAPE<b>')
    fig.update_layout(height=height, width=width, 
                      title_text="<b>Mean Absolute Percentage Error for " + "m-" 
                      + model_scores['Type'].values[0] + "</b>")
    
    return fig