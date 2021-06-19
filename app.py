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
myheading1 = 'Sentiment Analysis'
tabtitle = 'Sentiment Analysis'
githublink = 'https://github.com/fred-home/datascience-final'
notebookurl = 'https://github.com/fred-home/datascience-final/blob/master/analysis/final-project.ipynb'

# Load the processed data into Pandas DataFrame
df_analyzed_reviews = pd.read_csv('data/df_analyzed_reviews.csv')

# Data prep must be performed after loading from CVS file
df_analyzed_reviews['date'] = pd.to_datetime(df_analyzed_reviews['date'])

# Create a list of years between 2012-2020 since there no data for some parks before 2012
years = np.arange(2012, 2021)
parks = ['Florida', 'Japan', 'Singapore']

# Set initial selection of features to display when page loads
features = ['Singapore', 2019, 9]

# Initiate the app; must include '__name__' since it is used to locate root of project
# add the meta tagas to improve repsonsiveness on mobile devices
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.title = tabtitle

# Controls to place on page for feature selection by user
controls = dbc.Card(
    [
        html.H4('Features', className='text-center'),
        html.P('Change selections below to adjust the data used for plot.'),
        dbc.FormGroup(
            [
                dbc.Label('Park'),
                dcc.Dropdown(
                    id='park-drop',
                    options=[
                        {'label': park, 'value': park} for park in parks
                    ],
                    value=features[0],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Year'),
                dcc.Dropdown(
                    id='year-drop',
                    options=[
                        {'label': year, 'value': year} for year in years
                    ],
                    value=features[1],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Month'),
                dcc.Dropdown(
                    id='month-drop',
                    options=[
                        {'label': month, 'value': month} for month in range(1, 13)
                    ],
                    value=features[2],
                ),
            ]
        ),
    ],
    body=True,
)

# Set up the layout
app.layout = dbc.Container(
    [
        html.H1(myheading1, className='text-center'),
        html.Hr(),
        html.P('I used Natural Language Processing to analyze the text from reviews ' +
               'for Universal Studio Parks and then compare the sentiment of the review ' +
               'text to the actual rating given to the review.'),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Col(
                                dcc.Graph(
                                    id='figure-1',
                                    config={
                                        'modeBarButtonsToRemove': [
                                            'lasso2d', 'pan2d', 'select2d',
                                            'sendDataToCloud', 'toggleSpikelines', 'toImage',
                                            'autoScale2d', 'resetScale2d',
                                            'zoom2d', 'zoomIn2d', 'zoomOut2d'],
                                    },
                                ),
                                md=12),
                        ),
                        dbc.Row([
                                dbc.Col(md=3),
                                html.H5('',
                                        id='title-table',
                                        className='col-md-5'
                                        ),
                                dbc.Col(md=4),
                                ],
                                className='justify-content-center text-center',
                                ),
                        dbc.Row(
                            [
                                dbc.Col(md=3),
                                html.Div(
                                    id='div-table',
                                    className='col-md-5',
                                ),
                                dbc.Col(md=4),
                            ],
                            className='justify-content-center text-center',
                        ),
                    ],
                    md=9),
            ],
            align='center',
        ),
        html.Footer(
            [
                html.Div(
                    [
                        html.Small([
                            html.A(
                                'Jupyter Notebook',
                                href=notebookurl,
                            ),
                            ' on GitHub',
                        ]),
                        html.Small([
                            ', ',
                            html.A(
                                'source code for project',
                                href=githublink,
                            ),
                            ' on GitHub',
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


@ app.callback(
    [
        Output('figure-1', 'figure'),
        Output(component_id='title-table', component_property='children'),
        Output(component_id='div-table', component_property='children'),
    ],
    [Input('park-drop', 'value'),
     Input('year-drop', 'value'),
     Input('month-drop', 'value'),
     ])
def check_sentiment(park, year, month):

    try:
        # Year and Month parameter values are strings, convert to integers
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
        the_title = f'{park} - Review Sentiment - {month_name} {year}'
        table_title = f'{park} - {month_name} {year}'

        fig = go.Figure(data)

        fig.update_layout(
            xaxis=dict(title='Rating'),
            yaxis=dict(title='Total Reviews'),
            title=dict(text=the_title),
            title_x=0.45,      # shift title to the right to be closer to center
            title_y=0.99,      # Move plot title to be closer to top
            # shift legend down a small amount to make space for modebar
            legend=dict(title='Sentiment', yanchor='top', y=0.90),
            # mode: Compare data on hover (shows tags for all values at selected x position)
            hovermode='x',
            # reduce space around plot (top, botton, left, right) to use all space for the plot
            margin=dict(t=25, b=60, l=10, r=10),
        )

        return [fig, table_title, generate_table(summary_df)]
    except Exception as ex:
        return ["inadequate inputs", "inadequate inputs", "inadequate inputs...\n" + str(ex)]

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
