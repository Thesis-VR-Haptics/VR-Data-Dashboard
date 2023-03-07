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
            dbc.Col(dcc.RadioItems(id='controller', options=[' All ', ' Oculus Quest 2 Controllers ', ' Sense Glove ', ' Hi5 '],
                       value='All'), width = 4),
            dbc.Col(dcc.Dropdown(id="opt_dropdown",multi=False, style={'width': "100%"}), width = 3)
        ]),
        dbc.Row([dcc.Tabs([
            dcc.Tab(label='Exercise', children=[dcc.Graph(id="exercise_plot", style={'display': 'inline-block'}, figure={}),dcc.Graph(id="speed_plot", style={'display': 'inline-block'}, figure={})]),
            dcc.Tab(label='Training', children=[dcc.Graph(id="training_plot", style={'display': 'inline-block'}, figure={})])])])
    ])

    def getRunIDs(username, controller):
        #TODO: run_db en user_db van de mysql server halen
        user_db = pd.read_csv("user_csv.csv", delimiter=',')
        user_id = user_db.iloc[user_db.index[user_db['username'] == username].tolist()[0], 0]

        run_db = pd.read_csv("run_data_csv.csv", usecols=("run_id", "u_id", "controller","time_start"), delimiter=',')
        run_db = run_db[run_db['u_id']==user_id]
        run_db = run_db.iloc[::-1]
        list = []
        for i in range(len(run_db)):
            str=f'{run_db.iloc[i,2]}, {run_db.iloc[i,3]}'
            list.append(str)
        return list

    def get3DFig():
        o_db = pd.read_csv("data\QuestController_notJittery.csv")
        visualizer = Visualizer("data\QuestController_notJittery.csv")
        visualizer.initializeVectors(True)
        """
        x = o_db.iloc[:, 2]
        y = o_db.iloc[:, 3]
        z = o_db.iloc[:, 4]
        t = o_db.iloc[:, 0]
        """
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

    def getSpeedFig():
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Position',  'Speed (m/s)', 'Acceleration (m/s^2)'))

        visualizer = Visualizer("data\QuestController_notJittery.csv")
        visualizer.initializeVectors(True)

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
        return fig

    @app.callback([Output('opt_dropdown','options'),Output(component_id='exercise_plot', component_property='figure'),Output(component_id='speed_plot', component_property='figure')],
                  [Input('submit-button', 'n_clicks')],
                  [State('username', 'value'), State('controller', 'value')],
                  )
    def update_output(clicks, username_value, controller_value):
        opts = getRunIDs(username_value, controller_value)
        options = [{'label': opt, 'value': opt} for opt in opts]
        fig3D = get3DFig()
        figSpeed = getSpeedFig()
        if clicks is not None:
            return options, fig3D, figSpeed

    app.run_server(debug=False)