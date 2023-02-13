from dash import Dash, dcc, html, Input, Output, dash_table
from module.ma_model import MovingAverage
from module.plot_functions import plot_metrics
import os
import pandas as pd
import numpy as np


# load pre-processed data
train = pd.read_pickle('data/train.pkl')
test = pd.read_pickle('data/test.pkl')
forecast = pd.read_csv('data/forecast.csv')
forecast['Date'] = pd.to_datetime(forecast['Date'])
ma_metrics = None


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app = Dash(__name__)

ma_nav_style = {
    'borderBottom': '1px solid #d6d6d6',
    #'padding': '6px',
    'backgroundColor': '#e6ffe6',
}

tabs_styles = {
    'height': '44px',
    'fontSize': '14pt',
}

sub_tabs_styles = {
    'height': '44px',
    'fontSize': '14pt',
    'width': '700px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'backgroundColor': '#ccffcc',
}

sub_tab_style = {
    'backgroundColor': '#ccffeb',
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#009900',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
}

tab_content = {
    'backgroundColor': '#1a3300',
}

sub_tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#004d00',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
}

section_title = {
    'color': '#003300',
    'padding': '4px',
    'fontWeight': 'bold',
    'fontSize': '12pt'
}

col_divider_style = {
    'padding': '10px',
}

app.layout = html.Div([
    html.H1('Time Series Forecast: Walmart Sales (2010 - 2012)', style={'color':'#006600'}),
    html.Div([
        html.H4('Last Updated: 2/12/2023 by Ngoc Phan (', style={'display':'inline'}),
        dcc.Link(['LinkedIn'], title='LinkedIn', target="_blank", 
                 href='https://www.linkedin.com/in/nphan-usvn/', 
                 style={'display':'inline'}),
        html.H4(' | ', style={'display':'inline'}),
        dcc.Link(['GitHub'], title='GitHub', target="_blank", 
                 href='https://github.com/nphan20181/time_series_forecast', 
                 style={'display':'inline'}),
        html.H4(')', style={'display':'inline'}),
    ]), # end html.Div
    
    dcc.Tabs(
        children=[
            dcc.Tab(
                label='Moving Average',
                id='tab-MA',
                style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Div([
                            html.Table([
                                html.Tr([
                                    html.Td([
                                        html.Div('Model:', style={'fontWeight':'bold'})
                                    ]), # end html.Td
                                    html.Td([
                                        html.Div(id='ma-model-name', 
                                                 style={'color':'red', 
                                                        'fontWeight':'bold',
                                                        'backgroundColor':'#ffffcc'}), # end html.Div
                                    ]), # end html.Td
                                    html.Td(style={'width':'20px'}),
                                    html.Td([
                                        html.Div('MAPE:', style={'fontWeight':'bold'})
                                    ]), # end html.Td
                                    html.Td([
                                        html.Div(id='ma-model-MAPE', 
                                                 style={'color':'red', 
                                                        'fontWeight':'bold',
                                                        'backgroundColor':'#ffffcc'}), # end html.Div
                                    ]), # end html.Td
                                    html.Td(style={'width':'20px'}),
                                    html.Td([
                                        html.Div('Moving Window:', style={'fontWeight':'bold'})
                                    ]), # end html.Td
                                    html.Td([
                                        dcc.Input(id='ma-order', type='number', value=52, min=1, max=52, step=1)
                                    ]), # end html.Td
                                    html.Td(style={'width':'20px'}),
                                    html.Td([
                                        html.Div('Moving Type:', style={'fontWeight':'bold'})
                                    ]), # end html.Td
                                    html.Td([
                                        dcc.RadioItems(
                                            options=[
                                                {'label': 'Average', 'value': 'MA'},
                                                {'label': 'Median', 'value': 'MM'},
                                            ],
                                            value='MA', id='rdoMaType'
                                        ) # end dcc.RadioItems
                                    ]), # end html.Td
                                    html.Td(style={'width':'20px'}),
                                    html.Td([
                                        html.Div('Transform Method:', style={'fontWeight':'bold'})
                                    ]), # end html.Td
                                    html.Td([
                                        dcc.RadioItems(
                                            options=[
                                                {'label': 'None', 'value': ''},
                                                {'label': 'Logarithm', 'value': 'log'},
                                                {'label': 'Square Root', 'value': 'sqrt'},
                                            ],
                                            value='', id='rdoTransform'
                                        ) # end dcc.RadioItems
                                    ]), # end html.Td
                                ]), # end html.Tr
                            ]), # end html.Table
                        ], style=ma_nav_style), # end html.Div
                        
                        html.Table([
                            html.Tr([
                                html.Td([dcc.Graph(id='ma-fig-forecast')]), # end html.Td
                                html.Td(style=col_divider_style), # end html.Td
                                html.Td([dcc.Graph(id='ma-fig-errors')]), # end html.Td
                            ]), # end html.Tr
                        ]), # end html.Table -- second row
                        html.Table([
                            html.Tr([
                                html.Td([dcc.Graph(id='ma-fig-components')]), # end html.Td
                                html.Td(style={'width':'20px'}), # end html.Td
                                html.Td([dcc.Graph(id='ma-fig-seasonal-index')]), # end html.Td
                                html.Td(style={'width':'20px'}), # end html.Td
                                html.Td([
                                    dash_table.DataTable(
                                                id='ma-table-metrics',
                                                sort_action="native",
                                                filter_action='native',
                                                #fixed_rows={'headers': True},
                                                style_table={'width': '280px','height': '440px', 'overflow': 'auto'},
                                                style_cell={'height': 'auto', 'whiteSpace': 'normal', 'textAlign': 'left'},
                                                style_header={'backgroundColor':"paleturquoise", 'border': '1px solid green', 
                                                'textAlign': 'center', 'fontWeight': 'bold', 'height': 'auto', 'whiteSpace':'normal'},
                                                style_data={'backgroundColor':"lavender",'border': '1px solid white'},
                                                style_as_list_view=True,),    # end DataTable
                                ]), # end html.Td
                            ]), # end html.Tr
                        ]), # end html.Table -- 3rd row
                    ], style=tab_content), # end html.Div
                ] # end children
            ), # end dcc.Tab
            dcc.Tab(
                label='Exponential Smoothing',
                id='tab-Exponential',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
            dcc.Tab(
                label='Holt-Winters',
                id='tab-Holt',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
            dcc.Tab(
                label='Regression',
                id='tab-MLR',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
            dcc.Tab(
                label='ARMA',
                id='tab-ARMA',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
            dcc.Tab(
                label='SARIMAX',
                id='tab-SARIMAX',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
            dcc.Tab(
                label='RNN',
                id='tab-RNN',
                style=tab_style,
                selected_style=tab_selected_style,
            ),
        ], style=tabs_styles), # end Tabs
    html.Div(id='tabs-content-classes')
]) # end main Div

@app.callback(Output('ma-model-name', 'children'),
              Output('ma-model-MAPE', 'children'),
              Output('ma-fig-forecast', 'figure'),
              Output('ma-fig-errors', 'figure'),
              Output('ma-fig-components', 'figure'),
              Output('ma-fig-seasonal-index', 'figure'),
              Output('ma-table-metrics', 'data'),
              Output('ma-table-metrics', 'columns'),
              Input('ma-order', 'value'),
              Input('rdoMaType', 'value'),
              Input('rdoTransform', 'value'))
def render_content(ma_order, moving_type, transform):
    global train, test, ma_metrics
    
    # build m-MA model
    ma_model = MovingAverage(ma_order, transform, moving_type)
    ma_model.fit(train, test)
    ma_model.predict(forecast)
    
    # get model's name
    model_name = ma_model.model_name + ' (' + ma_model.ma_label + ')'
    
    # get evaluation metrics
    if ma_metrics is None:
        ma_metrics = ma_model.evaluate()
    else:
        ma_metrics = pd.concat([ma_metrics, ma_model.evaluate()], axis=0)
        ma_metrics.drop_duplicates(inplace=True, ignore_index=True)
    
    # prepare columns for data table
    metrics_cols = [{"name": i, "id": i} for i in ma_metrics.columns]
    
    # get mape score
    mape = str(np.round(ma_metrics[ma_metrics['Model'] == ma_model.ma_label]['MAPE'].values[0] * 100, 2)) + '%'
    
    return model_name, mape, ma_model.plot_forecast(width=900, height=500), ma_model.plot_errors(width=680, height=470), ma_model.plot_ma_components(), ma_model.plot_seasonal_indexes(), ma_metrics.to_dict('records'), metrics_cols

if __name__ == '__main__':
    app.run_server(debug=False)