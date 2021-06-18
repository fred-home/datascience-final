import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import calendar
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

from dash.dependencies import Input, Output, State

########### Define your variables ######
myheading1 = 'Universal Studios Reviews - Sentiment Analysis'
tabtitle = 'Sentiment Analysis'
sourceurl = 'https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/'
githublink = 'https://github.com/fred-home/datascience-final'

########### load the processed data into Pandas DataFrame ######
df_analyzed_reviews = pd.read_csv('data/df_analyzed_reviews2.csv')

# Data prep
df_analyzed_reviews['date'] = pd.to_datetime(df_analyzed_reviews['date'])

# Get list if years from the dataframe
#years = df_analyzed_reviews['date'].dt.year.unique()
#years = np.sort(years).tolist()

# Resatrict years to 2012-2020 since missing data from some parks before 2012
years = np.arange(2012, 2021)

# Initiate the app
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.title = tabtitle

# Controls to place on page
controls = dbc.Card(
    [
        html.H2('Features'),
        dbc.FormGroup(
            [
                dbc.Label("Universal Studios Park"),
                dcc.Dropdown(
                    id="park-drop",
                    options=[
                        {"label": park, "value": park} for park in ['Florida', 'Japan', 'Singapore']
                    ],
                    value="Florida",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Year"),
                dcc.Dropdown(
                    id="year-drop",
                    options=[
                        {"label": year, "value": year} for year in years
                    ],
                    value="2019",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Month"),
                dcc.Dropdown(
                    id="month-drop",
                    options=[
                        {"label": month, "value": month} for month in range(1, 13)
                    ],
                    value="6",
                ),
            ]
        ),
    ],
    body=True,
)

# Set up the layout
app.layout = dbc.Container(
    [
        html.H1(myheading1),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    [
                        dbc.Row(
                            dcc.Graph(id='figure-1'),
                        ),
                        dbc.Row([
                                html.H4('Sentiment DataFrame',
                                        className='col-md-5',
                                        ),
                                dbc.Col(className='col-md-3'),
                                ],
                                className='justify-content-center text-center',
                                ),
                        dbc.Row(
                            [
                                html.Div(
                                    id='div-table',
                                    className='col-md-5',
                                ),
                                dbc.Col(className='col-md-3'),
                            ],
                            className='justify-content-center text-center',
                        ),
                    ],
                    md=8),
            ],
            align="center",
        ),
        html.Footer(
            [
                html.Div(
                    [
                        html.Small([
                            html.A(
                                'Jupyter Notebook on GitHub',
                                href='https://github.com/fred-home/datascience-final/blob/master/analysis/final-project.ipynb',
                            ),
                        ]),
                    ],
                    className='fw-lighter p-2',
                ),
            ],
            className='fw-lighter fixed-bottom',
            style={'backgroundColor': 'rgba(0, 0, 0, 0.05)'}
        )
    ],
    fluid=True,
)

# Define Callback


@app.callback(
    [
        Output('figure-1', 'figure'),
        Output(component_id='div-table', component_property='children'),
    ],
    [Input('park-drop', 'value'),
     Input('year-drop', 'value'),
     Input('month-drop', 'value'),
     ])
def check_sentiment(park, year, month):

    try:
        # Must convert both Year and Month to integers
        year = int(year)
        month = int(month)

        df_park_month = df_analyzed_reviews[(df_analyzed_reviews['branch'] == park) &
                                            (df_analyzed_reviews['date'].dt.year == year) &
                                            (df_analyzed_reviews['date'].dt.month == month)].copy()

        summary_cols = ['neutral', 'negative', 'positive']
        summary_index = [5, 4, 3, 2, 1]

        summary_df = pd.DataFrame(columns=summary_cols,
                                  index=summary_index)

        for rating in summary_index:
            num_positive = df_park_month[(df_park_month['rating'] == rating)
                                         & (df_park_month['pos'] == 1)]['title'].count()
            num_negative = df_park_month[(df_park_month['rating'] == rating)
                                         & (df_park_month['neg'] == 1)]['title'].count()
            num_neutral = df_park_month[(df_park_month['rating'] == rating)
                                        & (df_park_month['neu'] == 1)]['title'].count()

            summary_df.loc[rating] = [num_neutral, num_negative, num_positive]

        data = []
        for col in summary_df.columns.to_list():
            data.append(
                go.Scatter(
                    x=summary_df.index,
                    y=summary_df[col],
                    name=col
                )
            )

        month_name = calendar.month_name[month]
        the_title = f'Universal Studios {park} Review Sentiment {month_name} {year}'

        fig = go.Figure(data)

        fig.update_layout(
            xaxis=dict(tickformat='0.0', title='Rating'),
            yaxis=dict(title='Total Reviews'),
            title=dict(text=the_title),
            legend=dict(title='Sentiment')
        )

        return [fig, generate_table(summary_df)]
    except Exception as ex:
        return "inadequate inputs", "inadequate inputs... " + str(ex)

# From Plotly Getting Started
def generate_table(dataframe, max_rows=10):
    mod_df = dataframe.copy().reset_index()
    cell_style = {'padding': '0.25rem 0.5rem'}
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th('', style=cell_style)
                        if col == 'index'
                        else html.Th(col, style=cell_style)
                        for col in mod_df.columns
                    ])
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Th(mod_df.iloc[i][col], style=cell_style)
                            if col == 'index'
                            else html.Td(mod_df.iloc[i][col], style=cell_style)
                            for col in mod_df.columns
                        ]) for i in (np.arange(min(len(mod_df), max_rows) - 1, -1, -1))
                        # rows list is in reverse order to get in ascending order in final table
                ])
        ],
        className='table small',
    )


# Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
