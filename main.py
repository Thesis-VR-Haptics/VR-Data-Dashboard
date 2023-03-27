import pandas as pd
import dash
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Visualizer import Visualizer

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

if __name__ == '__main__':
    app.layout = html.Div([
        dbc.Row(html.H1("VR Haptics Thesis Data", style={'text-align': 'center'})),
        dbc.Row([
            dbc.Col(dcc.Input(id='username', placeholder='Fill in username here', type='text'), width = 3),
            dbc.Col(html.Button(id='submit-button', type='submit', children='Submit'), width=2),
            dbc.Col(dcc.RadioItems(id='controller', options=[' All ', ' Oculus Quest 2 Controllers ', ' Sense Glove '],
                       value='All'), width = 4),
            dbc.Col(dcc.Dropdown(id="opt_dropdown",multi=False, style={'width': "100%"}), width = 3)
        ]),
        dbc.Row([dcc.Tabs([
            dcc.Tab(label='Apples Exercise', children=[dcc.Graph(id="exercise_plot", style={'display': 'inline-block'}, figure={}),
                                                dcc.Graph(id="speed_plot", style={'display': 'inline-block'}, figure={}),
                                                dbc.Row([
                dbc.Col(html.H3(children='')), dbc.Col(html.H3(children="1")),
                dbc.Col(html.H3(children="2")), dbc.Col(html.H3(children="3")),dbc.Col(html.H3(children="4")),dbc.Col(html.H3(children="Average"))]),
                                                dbc.Row([
                dbc.Col(html.H5(children="Smoothness")), dbc.Col(html.Div(id="sm1", children="-1")),
                dbc.Col(html.Div(id="sm2", children="-1")), dbc.Col(html.Div(id="sm3", children="-1")),dbc.Col(html.Div(id="sm4", children="-1")),dbc.Col(html.Div(id="smavg", children="-1"))]),
                                                dbc.Row([
                dbc.Col(html.H5(children="Average Speed")), dbc.Col(html.Div(id="avgs1", children="-1")),
                dbc.Col(html.Div(id="avgs2", children="-1")), dbc.Col(html.Div(id="avgs3", children="-1")),dbc.Col(html.Div(id="avgs4", children="-1")),dbc.Col(html.Div(id="avgavg", children="-1"))]),
                                                dbc.Row([
                dbc.Col(html.H5(children="Range")), dbc.Col(html.Div(id="r1", children="-1")),
                dbc.Col(html.Div(id="r2", children="-1")), dbc.Col(html.Div(id="r3", children="-1")),dbc.Col(html.Div(id="r4", children="-1"))])]),

            dcc.Tab(label='Training', children=[dbc.Row([
                dbc.Col(html.H3(children='''Exercise''')),dbc.Col(html.H3(children = "Average Smoothness")),dbc.Col(html.H3(children = "Time"))]),
                dbc.Row([
                dbc.Col(html.H5(children = "Apples")), dbc.Col(html.Div(id="appleAVG", children="-1")), dbc.Col(html.Div(id="appleTIME", children="-1"))]),
                dbc.Row([
                dbc.Col(html.H5(children="Drawing")), dbc.Col(html.Div(id="drawingAVG", children="-1")), dbc.Col(html.Div(id="drawingTIME", children="-1"))]),
                dbc.Row([
                dbc.Col(html.H5(children="Coffee")), dbc.Col(html.Div(id="coffeeAVG", children="-1")), dbc.Col(html.Div(id="coffeeTIME", children="-1"))])
            ]
            )])])])

    def getRunIDs(visualizer, username):
        #TODO: run_db en user_db van de mysql server halen
        #TODO: deze functie verplaatsen naar visualizer
       # user_db = pd.read_csv("user_csv.csv", delimiter=',')
        user_db = visualizer.getUserDB()
        user_id = user_db.iloc[user_db.index[user_db[3] == username].tolist()[0], 0]

        #run_db = pd.read_csv("run_data_csv.csv", usecols=("run_id", "u_id", "controller","time_start"), delimiter=',')
        run_db = visualizer.getRunsDB()
        run_db = run_db[run_db[1]==user_id]
        run_db = run_db.iloc[::-1]
        list = []
        for i in range(len(run_db)):
            str=f'{run_db.iloc[i,0]}'
            list.append(str)
        return list

    def get3DFig(visualizer):
        x = visualizer.x_axis_r
        y = visualizer.y_axis_r
        z = visualizer.z_axis_r
        t = visualizer.time

        trace = go.Scatter3d(
            x=x, y=y, z=z, mode='markers', marker=dict(
                size=3,
                color=t,  # set color to an array/list of desired values
                colorscale='Viridis'
            )
        )
        layout = go.Layout(title='3D movement')
        fig = go.Figure(data=[trace], layout=layout)
        return fig

    def getSpeedFig(visualizer):
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Position',  'Speed (m/s)', 'Acceleration (m/s^2)'))

        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.x_axis_r, mode = 'markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.y_axis_r, mode = 'markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.z_axis_r, mode = 'markers', marker=dict(size=2)),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.speed_vector, mode = 'markers', marker=dict(size=2)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.acceleration_vector, mode = 'markers',marker=dict(size=2)),
                      row=3, col=1)

        fig.update_layout(showlegend=False)
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Position', 'Speed (m/s)'))

        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.x_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.y_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.z_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=visualizer.time, y=visualizer.speed_vector, mode='markers', marker=dict(size=2)),
                      row=2, col=1)

        fig.update_layout(showlegend=False)

        return fig

    @app.callback([Output('opt_dropdown','options'),
                   Output(component_id='exercise_plot', component_property='figure'),
                   Output(component_id='speed_plot', component_property='figure'),
                   Output(component_id='drawingAVG', component_property='children'),
                   Output(component_id='coffeeAVG', component_property='children'),
                   Output(component_id='appleAVG', component_property='children'),
                   Output(component_id='drawingTIME', component_property='children'),
                   Output(component_id='appleTIME', component_property='children'),
                   Output(component_id='coffeeTIME', component_property='children'),
                   Output(component_id='sm1', component_property='children'),
                   Output(component_id='sm2', component_property='children'),
                   Output(component_id='sm3', component_property='children'),
                   Output(component_id='sm4', component_property='children'),
                   Output(component_id='smavg', component_property='children'),
                   Output(component_id='avgs1', component_property='children'),
                   Output(component_id='avgs2', component_property='children'),
                   Output(component_id='avgs3', component_property='children'),
                   Output(component_id='avgs4', component_property='children'),
                   Output(component_id='avgavg', component_property='children'),
                   Output(component_id='r1', component_property='children'),
                   Output(component_id='r2', component_property='children'),
                   Output(component_id='r3', component_property='children'),
                   Output(component_id='r4', component_property='children')
                   ],
                  ([Input('submit-button', 'n_clicks')],[Input('opt_dropdown','value')]),
                  [State('username', 'value'), State('controller', 'value')],
                  )
    def update_output(clicks, runChosen, username_value, controller_value):
        if clicks is not None:
            visualizer = Visualizer()
            opts = getRunIDs(visualizer, username_value)
            options = [{'label': opt, 'value': opt} for opt in opts]
            if runChosen[0] is not None:
                visualizer.setArraysFromDB(visualizer.getDataFromDB(runChosen[0]))
                r1,r2,r3,r4 = visualizer.getRangesDB(runChosen[0])
            else:
                visualizer.setArraysFromDB(visualizer.getDataFromDB(opts[0]))
                r1,r2,r3,r4 = visualizer.getRangesDB(opts[0])
           # visualizer.setArrays("data\QApplesFromDatabase")
            visualizer.initializeVectors(True)


            fig3D = get3DFig(visualizer)
            figSpeed = getSpeedFig(visualizer)
            appleavg, coffeavg, drawingavg, appleTime, coffeetime, drawingtime = visualizer.getValues()
            olsm, ilsm, orism, irsm, olavg, ilavg, oriavg, iravg = visualizer.sparcOnApples()
            smavg = np.round((olsm+ilsm+orism+irsm)/4,3)
            avgavg = np.round((olavg + ilavg + oriavg + iravg)/4,3)
            return options, fig3D, figSpeed, drawingavg, coffeavg, appleavg, drawingtime, appleTime, \
                coffeetime, olsm, ilsm, orism, irsm, smavg, olavg, ilavg, oriavg, iravg, avgavg,f"Right {r1}",f"Left {r2}",f"Up {r3}",f"Forward {r4}"

    app.run_server(debug=False)