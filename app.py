# final version
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
#import dash_html_components as html
from dash.dependencies import Input, Output
import yfinance as yf

from dash.dependencies import Input, Output, State
from dash import callback_context
#from plotly import tools
from plotly import subplots
from datetime import datetime
from datetime import timedelta
from dateutil.tz import gettz

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#%matplotlib inline

# Technical indicator library
import talib as ta

# Data Source
import yfinance as yf

from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
#from plotnine import *

# For time stamps
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

app = dash.Dash(__name__)
server = app.server
app.title = "Time Series Forecasting Stock Price Directional Movement"
app.config.suppress_callback_exceptions=True

# Loading data 
df = yf.download(tickers='TSLA', period='1y')
indicators = ['moving_average_trace','e_moving_average_trace','bollinger_trace','accumulation_trace','cci_trace','roc_trace','stoc_trace','mom_trace']

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#FFFFFF", ##f8f9fa
    "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("TradingViz", className="display-4"),
        html.Hr(),
        html.P(
            "Stock Directional Movement Forecast. You can refer to this "), html.A("website", href="https://www.nasdaq.com/market-activity/stocks/screener"), html.P("to see the tickers' symbols you would like to forecast.", className="lead"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Home", href="/home", active="exact")),
                dbc.NavItem(dbc.NavLink("Analysis", href="/analysis", active="exact")),
                dbc.NavItem(dbc.NavLink("Forecast", href="/forecast", active="exact")),
            ],
            vertical=True,
            pills=True,
            className="sidebar"
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)
#content = html.Div(id="page-content", style={"padding":"2rem"})

'''
def update_data(ticker, start_date, end_date):
    df2 = yf.download(tickers=ticker, start=start_date, end=end_date)
    return html.Div(
        children=[
            html
        ])
'''

# Display big numbers in readable format
def human_format(num):
    try:
        num = float(num)
        # If value is 0
        if num == 0:
            return 0
        else:
            return round(num, 4)

    except:
        return num
# Else value is a number
#if num < 1000000:
#    return num
#magnitude = int(math.log(num, 1000))
#mantissa = str(int(num / (1000 ** magnitude)))
#    return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


# Returns Top cell bar for header area
def get_top_bar_cell(cellTitle, cellValue):
    return html.Div(
        className="two-col",
        children=[
            html.P(className="p-top-bar", children=cellTitle),
            html.P(id=cellTitle, className="display-none", children=cellValue),
            html.P(children=human_format(cellValue)),
        ],
    )


# Returns HTML Top Bar for app layout
def get_top_bar(
    rmse=0, cc_lag0=0, spearmans=0, r2score=0
):
    return [
        get_top_bar_cell("RMSE", rmse),
        get_top_bar_cell("CC_Lag0", cc_lag0),
        get_top_bar_cell("Spearmans", spearmans),
        get_top_bar_cell("R2 score", r2score),
    ]

def get_top_fore_bar_cell(cellTitle, cellValue):
    return html.Div(
        className="two-col",
        children=[
            html.P(className="p-top-bar", children=cellTitle),
            html.P(id=cellTitle, className="display-none", children=cellValue),
            html.P(children=human_format(cellValue)),
        ],
    )

def get_top_fore_bar(
    rmse=0, r2score=0, next_move="-", return_=0
):
    return [
        get_top_bar_cell("RMSE", rmse),
        get_top_bar_cell("R2 score", r2score),
        get_top_bar_cell("Next Move", next_move),
        get_top_bar_cell("Return", return_),
    ]


# Main Chart Traces
def colored_bar_trace(filtered_data):
    return go.Ohlc(x=filtered_data.index, 
                    open=filtered_data["Open"], 
                    high=filtered_data["High"], 
                    low=filtered_data["Low"],
                    close=filtered_data['Close'],
                    showlegend=False,
                    name="colored bar",
                    )

def line_trace(filtered_data):
    return go.Scatter(x=filtered_data.index,
                    y=filtered_data["Close"],
                    mode="lines",
                    line=dict(color="#17B897"),
                    showlegend=False,
                    name="line")

def candlestick_trace(filtered_data):
    return go.Candlestick(x=filtered_data.index,
                            open=filtered_data["Open"],
                            high=filtered_data["High"],
                            low=filtered_data["Low"],
                            close=filtered_data["Close"],
                            showlegend=False,
                            name="candlestick",
                            )

def area_trace(filtered_data):
    return go.Scatter(x=filtered_data.index,
                        y=filtered_data["Close"],
                        showlegend=False,
                        fill="toself",
                        line=dict(color="#17B897"),
                        name="area")

def bar_trace(filtered_data):
    return go.Ohlc(x=filtered_data.index,
                    open=filtered_data["Open"],
                    high=filtered_data["High"],
                    low=filtered_data["Low"],
                    close=filtered_data["Close"],
                    increasing=dict(line=dict(color="#888888")),
                    decreasing=dict(line=dict(color="#888888")),
                    showlegend=False,
                    name="bar",
                    )

# Technical Indicators traces
def SMA(df, fig):
    df2 = df.rolling(window=5).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["Close"], mode="lines", showlegend=False, name="MA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig

# Moving average
def moving_average_trace(df, fig):
    df2 = df.rolling(window=5).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["Close"], mode="lines", showlegend=False, name="MA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Exponential moving average
def e_moving_average_trace(df, fig):
    df2 = df.rolling(window=20).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["Close"], mode="lines", showlegend=False, name="EMA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Bollinger Bands
def bollinger_trace(df, fig, window_size=10, num_of_std=5):
    price = df["Close"]
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    trace = go.Scatter(
        x=df.index, y=upper_band, mode="lines", showlegend=False, name="BB_upper"
    )

    trace2 = go.Scatter(
        x=df.index, y=rolling_mean, mode="lines", showlegend=False, name="BB_mean"
    )

    trace3 = go.Scatter(
        x=df.index, y=lower_band, mode="lines", showlegend=False, name="BB_lower"
    )

    fig.append_trace(trace, 1, 1)  # plot in first row
    fig.append_trace(trace2, 1, 1)  # plot in first row
    fig.append_trace(trace3, 1, 1)  # plot in first row
    return fig


# Accumulation Distribution
def accumulation_trace(df):
    df["Volume"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    )
    trace = go.Scatter(
        x=df.index, y=df["Volume"], mode="lines", showlegend=False, name="Accumulation"
    )
    return trace


# Commodity Channel Index
def cci_trace(df, ndays=5):
    TP = (df["High"] + df["Low"] + df["Close"]) / 3
    CCI = pd.Series(
        (TP - TP.rolling(window=10, center=False).mean())
        / (0.015 * TP.rolling(window=10, center=False).std()),
        name="cci",
    )
    trace = go.Scatter(x=df.index, y=CCI, mode="lines", showlegend=False, name="CCI")
    return trace


# Price Rate of Change
def roc_trace(df, ndays=5):
    N = df["Close"].diff(ndays)
    D = df["Close"].shift(ndays)
    ROC = pd.Series(N / D, name="roc")
    trace = go.Scatter(x=df.index, y=ROC, mode="lines", showlegend=False, name="ROC")
    return trace


# Stochastic oscillator %K
def stoc_trace(df):
    SOk = pd.Series((df["Close"] - df["Low"]) / (df["High"] - df["Low"]), name="SO%k")
    trace = go.Scatter(x=df.index, y=SOk, mode="lines", showlegend=False, name="SO%k")
    return trace


# Momentum
def mom_trace(df, n=5):
    M = pd.Series(df["Close"].diff(n), name="Momentum_" + str(n))
    trace = go.Scatter(x=df.index, y=M, mode="lines", showlegend=False, name="MOM")
    return trace

app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        sidebar,
        content,
        
    ],
)

# Callback for sidebar
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/home":
        
        return[
            html.Div(
                    children=[
                        html.P(children="Trading Viz💵", className="header-emoji"),
                        html.P(children=".", className="header-title"),
                        html.H2(children="Tommorow's Stock Directional Movement Forecast",className="header-title"),
                        html.P(
                            children="Time Series Forecasting Stock Price Directional Movement: Forecast the next direction! This application is created for Day Trader to decide when should they buy/sell the stock by observing the graph of the directional (UP/DOWN) forecasted by the model.",className="header-description",
                        ),
                    ],
                    className="header",
                ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(children="Ticker", className="menu-title"),
                            dcc.Input(id="ticker-filter", type="text", placeholder="TSLA",debounce=True, style={'marginRight':'10px'}, className="input"),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div(children="Chart Type", className="menu-title"),
                            dcc.Dropdown(
                                id="type-filter",
                                options=[
                                        {
                                         "label": "candlestick",
                                         "value": "candlestick_trace",
                                         },
                                         {"label": "line", "value": "line_trace"},
                                         {"label": "mountain", "value": "area_trace"},
                                         {"label": "bar", "value": "bar_trace"},
                                         {
                                          "label": "colored bar",
                                          "value": "colored_bar_trace",
                                          },
                                          ],
                                value="candlestick_trace",
                                clearable=False,
                                searchable=False,
                                className="dropdown",
                                ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Div(children="Indicators", className="menu-title"),
                            dcc.Dropdown(
                                id="indicators",
                                options=[
                                        {
                                            "label":i,"value":i
                                        }for i in indicators
                                    ],
                                value=[],
                                multi=True,
                                searchable=True,
                                className="dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children="Date Range",
                                className="menu-title"
                                ),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed="2010-01-01",
                                max_date_allowed=df.index.max().date(),
                                start_date=df.index.max().date(),
                                #start_date=df.index.min().date(),
                                end_date=df.index.max().date(),
                            ),
                        ]
                    ),
                    
                ],
                className="menu",
            ),
            html.P(
                    id="live_clock_MY",
                    #className="three-col",
                    className="time",
                    children=datetime.now().strftime("%H:%M:%S"),
                ),
            html.P(
                    id="live_clock_US",
                    #className="three-col",
                    className="time",
                    children=datetime.now().strftime("%H:%M:%S"),
                ),
            # Interval component for live clock
            dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0),
            # Interval component for graph updates
            dcc.Interval(id="i_tris", interval=1*60000, n_intervals=0),
            
            html.Div(
                children=[
                    #html.Div(
                    #    children = "Interval range", className="menu-title"),
                    html.Button('1D', id="1day", n_clicks=0),
                    html.Button('5D', id="5day", n_clicks=0),
                    html.Button('1M', id="1month", n_clicks=0),
                    html.Button('6M', id="6month", n_clicks=0),
                    html.Button('YTD', id="yday", n_clicks=0),
                    html.Button('1Y', id="1year", n_clicks=0),
                    html.Button('5Y', id="5year", n_clicks=0),
                    html.Button('Max', id="max", n_clicks=0),
                ],
                className="button"),

            html.Div(
                children=[
                    html.Div(
                        children=dcc.Graph(
                            id="multi_chart", config={'displayModeBar':'hover'}
                        ),
                        className="card",
                    )
                ],
                className="wrapper",
            )
        ]
    elif pathname == "/forecast":
        return [
                html.Div(
                    children=[
                        html.P(children="TradingViz 💵", className="header-emoji"),
                        html.P(children=".",className="header-title"),
                        html.H2(children="Forecasting Tab",className="header-title1"),
                        html.P(
                            children="Select the ticker you would like to forecast and see the result!",className="header-description",
                        ),
                    ],
                    className="header",
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(children="Enter the Ticker", className="menu-title"),
                                dcc.Input(id="input1", type="text", placeholder="TSLA",debounce=True, style={'marginRight':'10px'}, className="input"),
                            ]
                        ),
                        html.Div(
                            id="top_fore_bar", className="row div-top-fore-bar", children=get_top_fore_bar()
                        )
                    ],
                    className="menu",
                ),
                html.Div(
                    children=[
                        html.Div(dcc.Graph(id="graph")
                        )
                    ],
                    className="wrapper",
                )
                ]
    elif pathname == "/analysis":
        return [
                html.Div(
                    children=[
                        html.P(children="TradingViz 💵", className="header-emoji"),
                        html.P(children=".",className="header-title"),
                        html.H2(children="Analysis Tab",className="header-title1"),
                        html.P(
                            children="Select the ticker you would like to use to train the model and see the model's result!",className="header-description",
                        ),
                    ],
                    className="header",
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(children="Enter the Ticker", className="menu-title"),
                                dcc.Input(id="input1", type="text", placeholder="TSLA",debounce=True, style={'marginRight':'10px'}, className="input"),
                                
                            ]
                        ),
                        html.Div(
                            id="top_bar", className="row div-top-bar", children=get_top_bar()
                        ),
                    ],
                    className="menu",
                ),
            
                
                html.Div(
                    children=[
                        html.Div(dcc.Graph(id="train-test"))
                    ],
                    className="wrapper",
                )
                ]
    return[
        html.Div(
                children=[
                    html.P(children="TradingViz 💵", className="header-emoji"),
                    html.H2(children="Tommorow's Stock Directional Movement Forecast",className="header-title"),
                    html.P(
                        children="Time Series Forecasting Stock Price Directional Movement: Forecast the next direction! This application is created for Day Trader to decide when should they buy/sell the stock by observing the graph of the directional (UP/DOWN) forecasted by the model.",className="header-description",
                    ),
                ],
                className="header",
            ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Ticker", className="menu-title"),
                        dcc.Input(id="ticker-filter", type="text", placeholder="TSLA",debounce=True, style={'marginRight':'10px'}, className="input"),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Chart Type", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                    {
                                     "label": "candlestick",
                                     "value": "candlestick_trace",
                                     },
                                     {"label": "line", "value": "line_trace"},
                                     {"label": "mountain", "value": "area_trace"},
                                     {"label": "bar", "value": "bar_trace"},
                                     {
                                      "label": "colored bar",
                                      "value": "colored_bar_trace",
                                      },
                                      ],
                            value="candlestick_trace",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                            ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Indicators", className="menu-title"),
                        dcc.Dropdown(
                            id="indicators",
                            options=[
                                    {
                                        "label":i,"value":i
                                    }for i in indicators
                                ],
                            value=[],
                            multi=True,
                            searchable=True,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                            ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed="2010-01-01",
                            max_date_allowed=df.index.max().date(),
                            start_date=df.index.max().date(),
                            #start_date=df.index.min().date(),
                            end_date=df.index.max().date(),
                        ),
                    ]
                ),

            ],
            className="menu",
        ),
        html.P(
                id="live_clock_MY",
                #className="three-col",
                className="time",
                children=datetime.now().strftime("%H:%M:%S"),
            ),
        html.P(
                id="live_clock_US",
                #className="three-col",
                className="time",
                children=datetime.now().strftime("%H:%M:%S"),
            ),
        # Interval component for live clock
        dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0),
        # Interval component for graph updates
        dcc.Interval(id="i_tris", interval=1*60000, n_intervals=0),

        html.Div(
            children=[
                #html.Div(
                #    children = "Interval range", className="menu-title"),
                html.Button('1D', id="1day", n_clicks=0),
                html.Button('5D', id="5day", n_clicks=0),
                html.Button('1M', id="1month", n_clicks=0),
                html.Button('6M', id="6month", n_clicks=0),
                html.Button('YTD', id="yday", n_clicks=0),
                html.Button('1Y', id="1year", n_clicks=0),
                html.Button('5Y', id="5year", n_clicks=0),
                html.Button('Max', id="max", n_clicks=0),
            ],
            className="button"),

        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="multi_chart", config={'displayModeBar':'hover'}
                    ),
                    className="card",
                )
            ],
            className="wrapper",
        )
    ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

# Callback to update live clock MY
@app.callback(Output("live_clock_MY", "children"), [Input("interval", "n_intervals")])
def update_time(n):
    string = "Malaysia time: "+datetime.now().strftime("%H:%M:%S")
    return string

# Callback to update live clock US
@app.callback(Output("live_clock_US", "children"), [Input("interval", "n_intervals")])
def update_time(n):
    now_US = datetime.now(gettz('US/Eastern'))
    string = "US time: " +now_US.strftime("%H:%M:%S")
    return string

# Callback for graph in Home
@app.callback(Output("multi_chart", "figure"),
             [
                 Input("ticker-filter", "value"),
                 Input("type-filter", "value"),
                 Input("date-range", "start_date"),
                 Input("date-range", "end_date"),
                 Input("1day", "n_clicks"),
                 Input("5day", "n_clicks"),
                 Input("1month", "n_clicks"),
                 Input("6month", "n_clicks"),
                 Input("yday", "n_clicks"),
                 Input("1year", "n_clicks"),
                 Input("5year", "n_clicks"),
                 Input("max", "n_clicks"),
                 Input("indicators", "value"),
                 Input("i_tris", "n_intervals")
            ],)

def update_graph(ticker, type, start_date, end_date, btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, indicators, n):
    if ticker == None:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    
        if "1day" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='1d', interval = "1m")
        elif "5day" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='5d', interval = "5m")
        elif "1month" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='1mo', interval = "1d")
        elif "6month" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='6mo')
        elif "yday" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='ytd')
        elif "1year" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='1y')
        elif "5year" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='5y')
        elif "max" in changed_id:
            filtered_data = yf.download(tickers="TSLA", period='max')
        elif start_date == end_date:
            filtered_data = yf.download(tickers="TSLA", period='1d', interval = "1m")
        else:
            filtered_data = yf.download(tickers="TSLA", start=start_date, end=end_date)

        subplot_traces = [  # first row traces
            "accumulation_trace",
            "cci_trace",
            "roc_trace",
            "stoc_trace",
            "mom_trace",
        ]
        selected_subplots_studies = []
        selected_first_row_studies = []
        row = 1  # number of subplots

        if indicators:
            for study in indicators:
                if study in subplot_traces:
                    row += 1  # increment number of rows only if the study needs a subplot
                    selected_subplots_studies.append(study)
                else:
                    selected_first_row_studies.append(study)

        fig = subplots.make_subplots(
            rows=row,
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.12,
        )
        if start_date == end_date:
            fig.update_layout(
                    title = {
                        'text': f'TSLA live share price evolution',
                        'y':0.9,
                        'x':0.5,
                        'xanchor':'center',
                        'yanchor':'top'
                    }
            )  
        else:
            fig.update_layout(
                    title = {
                        'text': f'Stock Price of TSLA from {start_date} to {end_date}',
                        'y':0.9,
                        'x':0.5,
                        'xanchor':'center',
                        'yanchor':'top'
                    }
            )
        if "5day" in changed_id:
            fig.update_xaxes(
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
                ])
        else:
            fig.update_xaxes(
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    #dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
                ])

        # Add main trace (style) to figure
        fig.append_trace(eval(type)(filtered_data), 1, 1)

        # Add trace(s) on fig's first row
        for study in selected_first_row_studies:
            fig = eval(study)(filtered_data, fig)

        row = 1
        # Plot trace on new row
        for study in selected_subplots_studies:
            row += 1
            fig.append_trace(eval(study)(filtered_data), row, 1)
        return fig
    
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    
    if "1day" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='1d', interval = "1m")
    elif "5day" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='5d', interval = "5m")
    elif "1month" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='1mo', interval = "1d")
    elif "6month" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='6mo')
    elif "yday" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='ytd')
    elif "1year" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='1y')
    elif "5year" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='5y')
    elif "max" in changed_id:
        filtered_data = yf.download(tickers=ticker, period='max')
    elif start_date == end_date:
        filtered_data = yf.download(tickers=ticker, period='1d', interval = "1m")
    else:
        filtered_data = yf.download(tickers=ticker, start=start_date, end=end_date)

    subplot_traces = [  # first row traces
        "accumulation_trace",
        "cci_trace",
        "roc_trace",
        "stoc_trace",
        "mom_trace",
    ]
    selected_subplots_studies = []
    selected_first_row_studies = []
    row = 1  # number of subplots
    
    if indicators:
        for study in indicators:
            if study in subplot_traces:
                row += 1  # increment number of rows only if the study needs a subplot
                selected_subplots_studies.append(study)
            else:
                selected_first_row_studies.append(study)

    if row == 1:
        fig = subplots.make_subplots(
            rows=row,
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.12,
            specs=[[{"secondary_y": True}]],
        )
    else:
        fig = subplots.make_subplots(
            rows=row,
            shared_xaxes=True,
            shared_yaxes=True,
            cols=1,
            print_grid=False,
            vertical_spacing=0.12,
            specs=[[{"secondary_y": True}],[{"secondary_y": False}]],
        )
        
    if start_date == end_date:
        fig.update_layout(
                title = {
                    'text': f'{ticker} live share price evolution',
                    'y':0.9,
                    'x':0.5,
                    'xanchor':'center',
                    'yanchor':'top'
                }
        )  
    else:
        fig.update_layout(
                title = {
                    'text': f'Stock Price of {ticker} from {start_date} to {end_date}',
                    'y':0.9,
                    'x':0.5,
                    'xanchor':'center',
                    'yanchor':'top'
                }
        )
    if "5day" in changed_id:
        fig.update_xaxes(
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
            ])
    else:
        fig.update_xaxes(
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                #dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
            ])
    
    # Add main trace (style) to figure
    fig.add_trace(eval(type)(filtered_data), row=1, col=1, secondary_y=False)
    
    filtered_data['diff'] = filtered_data['Close'] - filtered_data['Open']
    filtered_data.loc[filtered_data['diff']>=0, 'color'] = 'mediumseagreen'
    filtered_data.loc[filtered_data['diff']<0, 'color'] = 'red'

    fig.add_trace(go.Bar(x=filtered_data.index, y=filtered_data['Volume'], name='Volume', marker={'color':filtered_data['color']}, opacity=0.5), row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[0,filtered_data['Volume'].max()*10],secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=True)
    fig.update_layout(xaxis_rangeslider_visible=False)  #hide range slider
    fig.update_layout(showlegend=False) #not showing legend
    # Add trace(s) on fig's first row
    for study in selected_first_row_studies:
        fig = eval(study)(filtered_data, fig)

    row = 1
    # Plot trace on new row
    for study in selected_subplots_studies:
        row += 1
        fig.append_trace(eval(study)(filtered_data), row, 1)

    return fig

# Callback for graph in analysis
@app.callback(Output("train-test", "figure"),
              Output("top_bar", "children"),
             [
                Input("input1", "value"),
            ],)
def display_traintest(ticker):
    if ticker == None:
        fig = go.Figure()
        return fig, get_top_bar(0,0,0,0)
    data = yf.download(tickers=ticker, period='10y')

    forecasting_step = 60

    for i in range(1, forecasting_step):
        data['Close-%i' % (i)] = data['Close'].shift(i)

    # Add  Trading Indicator
    # Add RSI(Relative Strength Index)
    n = 60 # windows size
    data['RSI'] = ta.RSI(np.array(data['Close']),timeperiod = n)

    # ATR
    data['ATR'] = ta.ATR(np.array(data['High']), np.array(data['Low']), np.array(data['Close']),timeperiod = n)
    data

    # Create a column by name, SMA and assign the SMA calculation to it
    data['SMA'] = data['Close'].rolling(window=n).mean()

    # Create a column by name, Corr and assign the calculation of correlation to it
    #data['Corr'] = data['Close'].rolling(window=n).corr(data['SMA'])

    # Creata a column by name, ADX and assigin the ADX calculation to it
    data['ADX'] = ta.ADX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Create a column by name, SAR and assign the SAR calculation to it
    data['SAR'] = ta.SAR(np.array(data['High']), np.array(data['Low']),
                        0.2,0.2)

    # Directional Movement Index
    data['DX'] = ta.DX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Commodity Channel Index
    data['CCI'] = ta.DX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Two window size parameters for calculating EMAs and period for smoothing the signal line
    # can make into buy sell signal (point the lines interchange)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = ta.MACD(np.array(data['Close']), fastperiod=3, slowperiod=20, signalperiod=3)

    # Slow Stochastic
    data['SlowK'], data['SlowD'] = ta.STOCH(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # Fast Stochastic
    data['FastK'], data['FastD'] = ta.STOCHF(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), fastk_period=5, fastd_period=3, fastd_matype=0)

    # Williams' %R
    data['WILLR'] = ta.WILLR(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    data['Signal'] = data['Close'].shift(-1)
    #data = data.dropna()
    dataset = data.drop(['Volume','Open','High','Low','Adj Close'], axis=1)

    ss= StandardScaler()
    scaled_data = ss.fit_transform(dataset)

    # FORECAST NEXT DAY
    forecast_1 = pd.DataFrame(scaled_data).iloc[-1].drop([75], axis=0)

    scaled_data = pd.DataFrame(scaled_data)
    scaled_data = scaled_data.dropna()
    scaled_data = np.array(scaled_data)

    X = scaled_data[:,:-1]
    y = scaled_data[:,-1]

    training_data_len = int(np.ceil( len(X) * .95 ))

    x_train = X[0:int(training_data_len), :]
    y_train = y[0:int(training_data_len),]
    x_test = X[int(training_data_len):, :]
    y_test = y[int(training_data_len):,]

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions_4 = lr.predict(x_test)
    r2 = r2_score(predictions_4,y_test)
    rmse_4 = np.sqrt(mean_squared_error(predictions_4,y_test))

    next_day = lr.predict(pd.DataFrame(forecast_1).T)
    
    # Plot the data
    train_4 = pd.DataFrame(y_train)
    valid_4 = pd.DataFrame(y_test)
    valid_4["index"] = 0
    valid_4["index"][0] = train_4.index[-1] + 1
    for i in range(1,len(valid_4)):
        valid_4["index"][i] = valid_4["index"][i-1] + 1
    valid_4.set_index("index", inplace=True)
    valid_4['Predictions'] = predictions_4
    CC = sm.tsa.stattools.ccf(np.ndarray.flatten(y_test), np.ndarray.flatten(predictions_4), unbiased=False)
    corr_4, _ = spearmanr(y_test, predictions_4)
    
    # visualize the data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_4.index,
                y=train_4[0], name='train', line=dict(color='blue', width=1.2)))

    fig.add_trace(go.Scatter(x=valid_4.index,
                    y=valid_4[0], name='test', line=dict(color='grey', width=1.2)))

    fig.add_trace(go.Scatter(x=valid_4.index,
                    y=valid_4['Predictions'], name='pred', line=dict(color='red', width=1.2)))

    # Add titles
    fig.update_layout(
        title='Linear Regression Model',
        yaxis_title='Stock Price USD per Shares (Standardized)')
    rmse = rmse_4
    cc_lag0 = CC[0]
    spearmans = corr_4
    r2score = r2

    return fig, get_top_bar(rmse, cc_lag0, spearmans, r2score)

# Callback for graph in forecast
@app.callback(Output("graph", "figure"),
              Output("top_fore_bar", "children"),
             [
                Input("input1", "value"),
            ],)
def display_chart(ticker):
    if ticker == None:
        fig = go.Figure()
        return fig, get_top_fore_bar(0,0,"-",0)
    data = yf.download(tickers=ticker, period='10y')

    forecasting_step = 60

    for i in range(1, forecasting_step):
        data['Close-%i' % (i)] = data['Close'].shift(i)

    # Add  Trading Indicator
    # Add RSI(Relative Strength Index)
    n = 60 # windows size
    data['RSI'] = ta.RSI(np.array(data['Close']),timeperiod = n)

    # ATR
    data['ATR'] = ta.ATR(np.array(data['High']), np.array(data['Low']), np.array(data['Close']),timeperiod = n)
    data

    # Create a column by name, SMA and assign the SMA calculation to it
    data['SMA'] = data['Close'].rolling(window=n).mean()

    # Create a column by name, Corr and assign the calculation of correlation to it
    #data['Corr'] = data['Close'].rolling(window=n).corr(data['SMA'])

    # Creata a column by name, ADX and assigin the ADX calculation to it
    data['ADX'] = ta.ADX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Create a column by name, SAR and assign the SAR calculation to it
    data['SAR'] = ta.SAR(np.array(data['High']), np.array(data['Low']),
                        0.2,0.2)

    # Directional Movement Index
    data['DX'] = ta.DX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Commodity Channel Index
    data['CCI'] = ta.DX(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    # Two window size parameters for calculating EMAs and period for smoothing the signal line
    # can make into buy sell signal (point the lines interchange)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = ta.MACD(np.array(data['Close']), fastperiod=3, slowperiod=20, signalperiod=3)

    # Slow Stochastic
    data['SlowK'], data['SlowD'] = ta.STOCH(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # Fast Stochastic
    data['FastK'], data['FastD'] = ta.STOCHF(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), fastk_period=5, fastd_period=3, fastd_matype=0)

    # Williams' %R
    data['WILLR'] = ta.WILLR(np.array(data['High']),np.array(data['Low']),
                        np.array(data['Close']), timeperiod=n)

    data['Signal'] = data['Close'].shift(-1)
    #data = data.dropna()
    dataset = data.drop(['Volume','Open','High','Low','Adj Close'], axis=1)

    ss= StandardScaler()
    scaled_data = ss.fit_transform(dataset)

    # FORECAST NEXT DAY
    forecast_1 = pd.DataFrame(scaled_data).iloc[-1].drop([75], axis=0)

    scaled_data = pd.DataFrame(scaled_data)
    scaled_data = scaled_data.dropna()
    scaled_data = np.array(scaled_data)

    X = scaled_data[:,:-1]
    y = scaled_data[:,-1]

    training_data_len = int(np.ceil( len(X) * .95 ))

    x_train = X[0:int(training_data_len), :]
    y_train = y[0:int(training_data_len),]
    x_test = X[int(training_data_len):, :]
    y_test = y[int(training_data_len):,]

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions_4 = lr.predict(x_test)
    r2 = r2_score(predictions_4,y_test)
    rmse_4 = np.sqrt(mean_squared_error(predictions_4,y_test))

    next_day = lr.predict(pd.DataFrame(forecast_1).T)
    
    # Plot the data
    pred_4 = pd.DataFrame(predictions_4)
    test_4 = pd.DataFrame(y_test)

    forecast = y_test[-1]
    forecast = np.append(forecast, next_day)
    forecast = pd.DataFrame(forecast)
    forecast["index"] = 0
    forecast["index"][0] = test_4.index[-1]
    forecast["index"][1] = test_4.index[-1] + 1
    forecast.set_index("index", inplace=True)

    # visualize the data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_4.index,
                    y=test_4[0], name='True value', line=dict(color='grey', width=1.2)))

    
    boo = forecast.iloc[1] > forecast.iloc[0]
    if boo[0]:
        fig.add_trace(go.Scatter(x=forecast.index,
                    y=forecast[0], name='forecast value', line=dict(color='green', width=1.2)))
        next_move = "UP"
    else:
        fig.add_trace(go.Scatter(x=forecast.index,
                    y=forecast[0], name='forecast value', line=dict(color='red', width=1.2)))
        next_move = "DOWN"

    # Add titles
    fig.update_layout(
        title='Linear Regression Model',
        yaxis_title='Stock Price USD per Shares (Standardized)')
    
    return_ = (forecast[0][forecast.index[1]]-forecast[0][forecast.index[0]])/forecast[0][forecast.index[0]]
    
    return fig, get_top_fore_bar(rmse_4, r2, next_move, return_)
'''
# Callback to update Top Bar values
@app.callback(Output("top_bar", "children"), [Input("orders", "children")])
def update_top_bar(orders):
    if orders is None or orders is "[]":
        return get_top_bar()

    orders = json.loads(orders)
    open_pl = 0
    balance = 50000
    free_margin = 50000
    margin = 0

    for order in orders:
        if order["status"] == "open":
            open_pl += float(order["profit"])
            conversion_price = (
                1 if order["symbol"][:3] == "USD" else float(order["price"])
            )
            margin += (float(order["volume"]) * 100000) / (200 * conversion_price)
        else:
            balance += float(order["profit"])

    equity = balance - open_pl
    free_margin = equity - margin
    margin_level = "%" if margin == 0 else "%2.F" % ((equity / margin) * 100) + "%"
    equity = "%.2F" % equity
    balance = "%.2F" % balance
    open_pl = "%.2F" % open_pl
    free_margin = "%.2F" % free_margin
    margin = "%2.F" % margin

    return get_top_bar(balance, equity, margin, free_margin, margin_level, open_pl)
'''
if __name__ == "__main__":
    from waitress import serve
    serve(app)
