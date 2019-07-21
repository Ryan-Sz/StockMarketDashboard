import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input,Output,State
import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import textwrap as tw
import dash_auth

USERNAME_PASSWORD_PAIRS = [['username', 'password']]

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

def stock_monte_carlo (start_price, days, mu, sigma) :
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in range(1, days) :
        drift[x] = mu*deltat
        shock[x] = np.random.normal(loc=0.0, scale=sigma*np.sqrt(deltat))
        price[x] = price[x-1] + price[x-1]*(drift[x] + shock[x])
    return price

#bg_colour = '#CEDFD9'
bg_colour = '#111111'
stockchart_color = '#FF5D44'
font_colour = '#2B2621'
font_colour = '#CED3DC'
#font_family = 'Courier New, monospace'
font_family = 'Sans-Serif'
font_size = 15

deltat = 1
days = 20
app = dash.Dash()
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

nsdq = pd.read_csv("NASDAQcompanylist.csv")
nsdq.set_index('Symbol', inplace=True)
options = []

for tic in nsdq.index :
    mydict = {}
    mydict['label'] = str(nsdq.loc[tic]["Name"]) + ' ' + tic
    mydict['value'] = tic
    options.append(mydict)

app.layout = html.Div([
                html.Div([
                    html.Div([html.H1('Ryan\'s Stock Price Dashboard')
                    ], style={'display':'inline-block',
                            'padding':10, 'font-family': font_family}),
                    html.Div([html.Img(src=encode_image('wolfwallstreet.png'), height=150)
                    ], style={'float':'right', 'display':'inline-block',
                            'padding':0})
                ], style={'backgroundColor':bg_colour,'fontColor': font_colour, 'fontFamily':font_family,
                        'padding':0, 'height':120}),

                html.Div([
                html.Div([
                    html.H3('1 - Choose a stock symbol: ', style={'paddingRight':'30px', 'color':font_colour}),
                    dcc.Dropdown(id='my_ticker_symbol',
                            options = options,
                            value='TSLA'
                            #multi=True
                )], style={'backgroundColor':bg_colour,'fontColor': 'black', 'color':'black', 'fontFamily':font_family, 'display':'inline-block', 'verticalAlign':'top', 'width':'30%',
                        'padding':10}),

                html.Div([html.H3('2 - Select a start and end date'),
                    dcc.DatePickerRange(id='my_date_picker',
                                        min_date_allowed=dt.datetime(2015,1,1),
                                        max_date_allowed=dt.datetime.today(),
                                        start_date=dt.datetime(2018,1,1),
                                        end_date=dt.datetime.today()
                                        )
                ], style={'backgroundColor':bg_colour,'fontColor': font_colour, 'fontFamily':font_family,'display':'inline-block',
                        'padding':10}),

                html.Div([
                    html.H3('3 - Enter the number of simulations (1-500) and hit submit'),
                    dcc.Input(id='n_runs_input',
                              placeholder='#',
                              type='number',
                              max=500,
                              min=1
                              ),
                    html.Button(id='submit-button',
                                n_clicks=0,
                                children='Submit',
                                style={'fontSize':24, 'marginLeft':'40px'})
                    # html.H3('Enter the number of simulations (1-500) and hit submit')
                ], style={'backgroundColor':bg_colour,'fontColor': font_colour, 'fontFamily':font_family, 'display':'inline-block',
                        'padding':10, 'verticalAlign':'top'})

                ]),

                html.Hr([]),

                html.Div([
                    dcc.Graph(id='my_graph',
                            figure={'data':[{'x':[1,2], 'y':[3,1]}],
                            #'layout':{'title':'Closing Price',
                            'layout':go.Layout(title='Closing Price',
                                                yaxis=dict(title='Price / $'),
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                font=dict(family=font_family, size=font_size, color=font_colour)
                                                )
                            },
                            style={'width':'75%', 'height':400})
                ], style={'backgroundColor':bg_colour,'fontColor': font_colour,
                        'padding':0}),

                html.Div([
                        html.Div([
                                dcc.Graph(id='pathGraph',
                                        figure={'data':[{'x':[1,4], 'y':[3,1]}],
                                                'layout':go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                                                    plot_bgcolor='rgba(0,0,0,0)',
                                                                    font=dict(family=font_family, size=font_size, color=font_colour)
                                                                    )
                                        })
                        ], style={'width':'49%', 'display':'inline-block'}),
                        html.Div([
                                dcc.Graph(id='mcGraph',
                                        figure={'data':[{'x':[1,2], 'y':[3,1]},],
                                                'layout':go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                                                    plot_bgcolor='rgba(0,0,0,0)',
                                                                    font=dict(family=font_family, size=font_size, color=font_colour)
                                                                    )
                                        })
                        ], style={'backgroundColor':bg_colour,'fontColor': font_colour,'width':'49%', 'display':'inline-block', 'float':'right'})
                ], style={'backgroundColor':bg_colour,'fontColor': font_colour,'width':'100%'}),

                html.Div([
                dcc.Markdown(tw.dedent('''
                        This dashboard uses [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) to simulate stock price movement.

                        The maximum number of simulations is capped at 500, although many thousands are needed to make the distribution smooth.

                        The stock price data is from [IEX](https://iextrading.com/apps/stocks/) and pulled in using [Pandas DataReader](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#iex)
                        ''')
                )
                ], style={'color':'lime', 'padding':10}),

                html.Div(id='json_df', style={'display':'none'}) # Hidden df for intermediate step!

], style={'backgroundColor':bg_colour,'fontColor': font_colour, 'color': font_colour})

@app.callback(Output('json_df', 'children'),
            [Input('submit-button', 'n_clicks')],
            [State('my_ticker_symbol', 'value'),
            State('my_date_picker', 'start_date'),
            State('my_date_picker', 'end_date'),
            State('n_runs_input', 'value')
    ])
def update_mc_graph(n_clicks, stock_ticker, start_date, end_date, n_runs) :
    start = dt.datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = dt.datetime.strptime(end_date[:10], '%Y-%m-%d')
    df = web.DataReader(stock_ticker, 'yahoo', start, end)

    rets = df['Close'].pct_change().dropna()

    mu = rets.mean()
    sigma = rets.std()
    start_price = df['Close'].iloc[len(df)-1]
    runs = n_runs

    columns = list(range(days))
    columns[0] = stock_ticker
    df_sim = pd.DataFrame(columns=columns)

    for run in range(runs) :
        df_sim.loc[run] = stock_monte_carlo(start_price, days, mu, sigma)

    return df_sim.to_json(orient='split')

@app.callback(Output('pathGraph', 'figure'),
            [Input('json_df', 'children')
    ])
def update_paths(json_df) :
    df_sim = pd.read_json(json_df, orient='split')
    stock_ticker = df_sim.columns[0]
    df_sim.columns.values[0] = 0
    data = []
    for i in range(len(df_sim)) :
        X = df_sim.iloc[i]
        data.append(go.Scatter(y=X,
                                mode='lines',
                                showlegend=False,
                                opacity=0.6)
                    )
    layout=go.Layout(title=stock_ticker + " Simulation Paths",
                        xaxis=dict(title='Trading Days'),
                        yaxis=dict(title='Price / $'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family=font_family, size=font_size, color=font_colour)
                    )
    plotly_fig = go.Figure(data=data, layout=layout)

    return plotly_fig

@app.callback(Output('mcGraph', 'figure'),
            [Input('json_df', 'children')
    ])
def update_hist(json_df) :
    df_sim = pd.read_json(json_df, orient='split')
    stock_ticker = df_sim.columns[0]
    simulations = np.zeros(len(df_sim))
    for i in range(len(df_sim)) :
        simulations[i] = df_sim.iloc[i][days-1]
    q = np.percentile(simulations, 5)
    m = np.percentile(simulations, 50)

    data = [go.Histogram(x=simulations,
                        histnorm = 'probability',
                        showlegend=False,
                        nbinsx=50
                        ),
            go.Scatter(x=[q,q],
                        y=[0,0.1],
                        mode='lines',
                        name='Value at Risk (5%%) $%.2f' %q
                        ),
            go.Scatter(x=[m,m],
                        y=[0,0.1],
                        mode='lines',
                        name='Median Price $%.2f' % m
                        )
            ]
    layout = go.Layout(title="Probability Distribution of %s Stock Price after %d Trading Days" % (stock_ticker,days),
                        yaxis=dict(range=[0,0.1]),
                        xaxis=dict(title='Price / $'),
                        bargap=0.01,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family=font_family, size=font_size, color=font_colour)
                        )
    plotly_fig = go.Figure(data=data, layout=layout)

    return plotly_fig

@app.callback(Output('my_graph', 'figure'),
            [Input('submit-button', 'n_clicks')],
            [State('my_ticker_symbol', 'value'),
            State('my_date_picker', 'start_date'),
            State('my_date_picker', 'end_date')
    ])
def update_graph(n_clicks, stock_ticker, start_date, end_date) :
    start = dt.datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = dt.datetime.strptime(end_date[:10], '%Y-%m-%d')
    df = web.DataReader(stock_ticker, 'yahoo', start, end)
    traces = [{'x':df.index, 'y':df['Close'], 'name':stock_ticker,
            'line':dict(color = stockchart_color, width = 3)}]
    fig = {'data': traces,
            'layout' : go.Layout(title=stock_ticker + ' Closing Price',
                                yaxis=dict(title='Price / $'),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family=font_family, size=font_size, color=font_colour)
                                )
    }
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
