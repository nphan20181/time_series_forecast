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

def plot_metrics(scores_df, width=600, height=430):
    fig = px.parallel_categories(scores_df, color="MAPE",
                                 dimensions=['Model', 'MAPE', 'MAD', 'RMSE'],
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 color_continuous_midpoint=2)
    fig.update_layout(title_text="<b>Model Evaluation on Test Data</b>", width=width, height=height)
    
    return fig