import dash
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from Visualizer import Visualizer

external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
image_path_lvl1 = "assets/lvl1.png"
image_path_lvl2 = "assets/lvl2.png"
image_path_lvl3 = "assets/lvl3.png"

if __name__ == '__main__':
    app.layout = html.Div([
        dbc.Row(html.H1("VR Haptics Thesis Data", style={'text-align': 'center','margin-left':'7px', 'margin-top':'7px'}),className="h-10"),
        dbc.Row([
            dbc.Col(dcc.Input(id='username', placeholder='Fill in username here', type='text',style={'margin-left':'7px', 'margin-top':'7px'}), width = 3),
            dbc.Col(html.Button(id='submit-button', type='submit', children='Submit'), width=2),
            dbc.Col(dcc.RadioItems(id='controller', options=[' All ', ' Oculus Quest 2 Controllers ', ' Sense Glove '],
                       value='All'), width = 4),
            dbc.Col(dcc.Dropdown(id="opt_dropdown",multi=False, style={'width': "100%"}), width = 3)
        ],className="h-10"),
        dbc.Row(
            [dcc.Tabs([
                dcc.Tab(label='Kitchen Exercise', children=
                    [dcc.Tabs([
                        dcc.Tab(label='Level 3', children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="exercise_plot", style={'margin-left':'7px', 'margin-top':'3px', 'template' : 'pulse'}, figure={})),
                                dbc.Col(dcc.Graph(id="speed_plot", style={'margin-left':'7px', 'margin-top':'7px'}, figure={})),
                                dbc.Col(html.Img(src = image_path_lvl3, style={'margin-left':'7px', 'margin-top':'3px'}))
                            ]),
                            dbc.Row([
                                dbc.Col(html.H3(children='',style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.H3(children="1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.H3(children="2",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.H3(children="3",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.H3(children="4",style={'margin-left':'7px', 'margin-top':'7px'})),dbc.Col(html.H3(children="Average",style={'margin-left':'7px', 'margin-top':'7px'}))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Smoothness",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="sm1", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.Div(id="sm2", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="sm3", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.Div(id="sm4", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),dbc.Col(html.Div(id="smavg", children="-1",style={'margin-left':'7px', 'margin-top':'7px'}))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Average Speed (m/s)",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="avgs1", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.Div(id="avgs2", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="avgs3", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.Div(id="avgs4", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),dbc.Col(html.Div(id="avgavg", children="-1",style={'margin-left':'7px', 'margin-top':'7px'}))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Range",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="r1", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),
                                dbc.Col(html.Div(id="r2", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})), dbc.Col(html.Div(id="r3", children="-1",style={'margin-left':'7px', 'margin-top':'7px'})),dbc.Col(html.Div(id="r4", children="-1",style={'margin-left':'7px', 'margin-top':'7px'}))
                            ]),
                            dbc.Row([
                                dbc.Col(html.H2(children="Movement Quality Score", id = "totalScoreLvl3", style={'margin-left': '7px', 'margin-top': '7px'})),
                                dbc.Col(html.H2(children="Objects Completed", id = "objslvl3", style={'margin-left': '7px', 'margin-top': '7px'})),
                            ])
                        ]),
                        dcc.Tab(label='Level 2', children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="exercise_plot_lvl2", style={'display': 'inline-block'}, figure={})),
                                dbc.Col(dcc.Graph(id="speed_plot_lvl2", style={'display': 'inline-block'}, figure={})),
                                dbc.Col(html.Img(src=image_path_lvl2, style={'margin-left': '7px', 'margin-top': '3px'}))
                            ]),
                            dbc.Row([
                                dbc.Col(html.H3(children='')), dbc.Col(html.H3(children="1")),
                                dbc.Col(html.H3(children="2")), dbc.Col(html.H3(children="Average"))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Smoothness")), dbc.Col(html.Div(id="sm1_lvl2", children="-1")),
                                dbc.Col(html.Div(id="sm2_lvl2", children="-1")),
                                dbc.Col(html.Div(id="smavg_lvl2", children="-1"))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Average Speed (m/s)")),
                                dbc.Col(html.Div(id="avgs1_lvl2", children="-1")),
                                dbc.Col(html.Div(id="avgs2_lvl2", children="-1")),
                                dbc.Col(html.Div(id="avgavg_lvl2", children="-1"))]),
                            dbc.Row([
                                dbc.Col(html.H2(children="Movement Quality Score", id="totalScoreLvl2", style={'margin-left': '7px', 'margin-top': '7px'})),
                                dbc.Col(html.H2(children="Objects Completed", id = "objslvl2", style={'margin-left': '7px', 'margin-top': '7px'})),
                            ])
                        ]),
                        dcc.Tab(label='Level 1', children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="exercise_plot_lvl1", style={'display': 'inline-block'}, figure={})),
                                dbc.Col(dcc.Graph(id="speed_plot_lvl1", style={'display': 'inline-block'}, figure={})),
                                dbc.Col(html.Img(src=image_path_lvl1, style={'margin-left': '7px', 'margin-top': '3px'}))
                            ]),
                            dbc.Row([
                                dbc.Col(html.H3(children='')), dbc.Col(html.H3(children="1")),
                                dbc.Col(html.H3(children="2")), dbc.Col(html.H3(children="Average"))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Smoothness")),
                                dbc.Col(html.Div(id="sm1_lvl1", children="-1")),
                                dbc.Col(html.Div(id="sm2_lvl1", children="-1")),
                                dbc.Col(html.Div(id="smavg_lvl1", children="-1"))]),
                            dbc.Row([
                                dbc.Col(html.H5(children="Average Speed (m/s)")),
                                dbc.Col(html.Div(id="avgs1_lvl1", children="-1")),
                                dbc.Col(html.Div(id="avgs2_lvl1", children="-1")),
                                dbc.Col(html.Div(id="avgavg_lvl1", children="-1"))
                            ]),
                            dbc.Row([
                                dbc.Col(html.H2(children="Movement Quality Score", id="totalScoreLvl1", style={'margin-left': '7px', 'margin-top': '7px'})),
                                dbc.Col(html.H2(children="Objects Completed", id = "objslvl1", style={'margin-left': '7px', 'margin-top': '7px'}))
                            ])
                        ])
                    ])
                    ]
                ),
                dcc.Tab(label='Painting Exercise', children=[
                    dbc.Row([
                        dbc.Col(width=3),
                        dbc.Col(dcc.Graph(id="painting1", style={'margin-left': '7px', 'margin-top': '3px'}, figure={})),
                        dbc.Col(dcc.Graph(id="painting2", style={'margin-left': '7px', 'margin-top': '7px'}, figure={})),
                        dbc.Col(dcc.Graph(id="painting3", style={'margin-left': '7px', 'margin-top': '7px'}, figure={}))
                    ]),
                    dbc.Row([
                        dbc.Col(html.H3(children='Accuracy', style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id = "squareAcc", children ="", style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id = "houseAcc", children="", style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id = "smileyAcc", children="", style={'margin-left': '7px', 'margin-top': '7px'}))
                    ]),
                    dbc.Row([
                        dbc.Col(html.H3(children='Smoothness', style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id = "squareSm" , children="", style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id ="houseSm", children = "", style={'margin-left': '7px', 'margin-top': '7px'})),
                        dbc.Col(html.H3(id="smileySm", children="", style={'margin-left': '7px', 'margin-top': '7px'}))
                    ]),
                    dbc.Row([
                        dbc.Col(html.H2(children="Movement Quality Score", id="totalScoreLvl4",
                                        style={'margin-left': '7px', 'margin-top': '7px'})),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="rainbow", style={'margin-left': '7px', 'margin-top': '3px'}, figure={})),
                        dbc.Col(dcc.Graph(id="rainbowsm", style={'margin-left': '7px', 'margin-top': '3px'}, figure={}))
                    ])
                ]),
                dcc.Tab(label='Training Progress', children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="progressFig", style={'margin-left': '7px', 'margin-top': '3px'}, figure={}))
                    ])
                ])
            ])]
        ,className="h-80")],
    style={"height": "100vh"})

    def getRunIDs(visualizer, username):
        user_db = visualizer.getUserDB()
        user_id = user_db.iloc[user_db.index[user_db[3] == username].tolist()[0], 0]
        run_db = visualizer.getRunsDB()
        run_db = run_db[run_db[1]==user_id]
        run_db = run_db.iloc[::-1]
        list = []
        times = []
        for i in range(len(run_db)):
            str=f'{run_db.iloc[i,0]}'
            list.append(str)
        for i in range(len(run_db)):
            str=f'{run_db.iloc[i,3]}'
            times.append(str)
        return list, times

    def getRBProgressFig(visualizer):
        e, f, g, h, i = visualizer.sparcOnRainbow()

        fig = make_subplots(rows=1, cols=1, subplot_titles="Movement")

        fig.add_trace(go.Scatter(y=[e,f,g,h,i], mode='markers', marker=dict(size=7, color = [1,2,3,4,5],colorscale='Viridis')),
                      row=1, col=1)
        fig.update_layout(template="none")

        return fig

    def get3DFig(visualizer, part):
        if(part != 7):
            x,y,z,t, color = visualizer.getAxesForPart(part)
        else:
            x, y, z, t, color = visualizer.getAxesForRainbow()
        trace = go.Scatter3d(
            x=x, y=y, z=z, mode='markers', marker=dict(
                size=3,
                color=color,  # set color to an array/list of desired values
                colorscale='Viridis'
            )
        )
        layout = go.Layout()
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(template="none")

        return fig

    def getSpeedFig(visualizer, part):
        ol = visualizer.original_db[(visualizer.original_db[13] == part)].reset_index(drop=True)
        olVisualizer = Visualizer()
        olVisualizer.setArraysFromDB(db=ol)
        olVisualizer.initializeVectors(True)
        color = olVisualizer.original_db.iloc[:, -1]
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Position', 'Speed (m/s)'))

        fig.add_trace(go.Scatter(x=olVisualizer.time, y=olVisualizer.x_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=olVisualizer.time, y=olVisualizer.y_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=olVisualizer.time, y=olVisualizer.z_axis_r, mode='markers', marker=dict(size=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=olVisualizer.time, y=olVisualizer.speed_vector, mode='markers', marker=dict(size=2, color = color, colorscale ="Viridis")),
                      row=2, col=1)

        fig.update_layout(showlegend=False)
        fig.update_layout(template="none")

        return fig

    def getProgressFigs(opts):
        smoothnessScores = []
        rangesleft = []
        rangesright = []
        rangesfront = []
        rangesup = []

        for i in opts:
            visualizer = Visualizer()
            visualizer.setArraysFromDB(visualizer.getDataFromDB(i))
            r1, r2, r3, r4 = visualizer.getRangesDB(i)
            visualizer.initializeVectors(True)
            smoothnessScores.append(visualizer.getAverageSmoothness())
            rangesleft.append(r1)
            rangesright.append(r2)
            rangesfront.append(r3)
            rangesup.append(r4)

        progressfig = make_subplots(rows=1, cols=2, subplot_titles=('Movement Quality (%)', 'Range (m)'))
        progressfig.add_trace(go.Scatter(y=smoothnessScores, showlegend=False, mode='lines+markers',marker=dict(size=7, color=[1, 2, 3, 4, 5], colorscale='Viridis')),row=1, col=1)

        progressfig.add_trace(go.Scatter(y=rangesleft, name="left", mode='lines+markers', marker=dict(size=4)), row=1, col=2)
        progressfig.add_trace(go.Scatter(y=rangesright, name="right", mode='lines+markers', marker=dict(size=4)),row=1, col=2)
        progressfig.add_trace(go.Scatter(y=rangesfront, name="front", mode='lines+markers', marker=dict(size=4)),row=1, col=2)
        progressfig.add_trace(go.Scatter(y=rangesup, name="up", mode='lines+markers', marker=dict(size=4)),row=1, col=2)

        progressfig.update_layout(template="none")

        return progressfig

    def emptyReturn():
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1

    @app.callback([Output('opt_dropdown','options'),
                   Output(component_id='exercise_plot', component_property='figure'),
                   Output(component_id='progressFig', component_property='figure'),
                   Output(component_id='speed_plot', component_property='figure'),
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

                   Output(component_id='totalScoreLvl3', component_property='children'),
                   Output(component_id='objslvl3', component_property='children'),
                   Output(component_id='r1', component_property='children'),
                   Output(component_id='r2', component_property='children'),
                   Output(component_id='r3', component_property='children'),
                   Output(component_id='r4', component_property='children'),

                   Output(component_id='sm1_lvl2', component_property='children'),
                   Output(component_id='sm2_lvl2', component_property='children'),
                   Output(component_id='smavg_lvl2', component_property='children'),
                   Output(component_id='avgs1_lvl2', component_property='children'),
                   Output(component_id='avgs2_lvl2', component_property='children'),
                   Output(component_id='avgavg_lvl2', component_property='children'),
                   Output(component_id='exercise_plot_lvl2', component_property='figure'),
                   Output(component_id='speed_plot_lvl2', component_property='figure'),
                   Output(component_id='totalScoreLvl2', component_property='children'),
                   Output(component_id='objslvl2', component_property='children'),

                   Output(component_id='sm1_lvl1', component_property='children'),
                   Output(component_id='sm2_lvl1', component_property='children'),
                   Output(component_id='smavg_lvl1', component_property='children'),
                   Output(component_id='avgs1_lvl1', component_property='children'),
                   Output(component_id='avgs2_lvl1', component_property='children'),
                   Output(component_id='avgavg_lvl1', component_property='children'),
                   Output(component_id='exercise_plot_lvl1', component_property='figure'),
                   Output(component_id='speed_plot_lvl1', component_property='figure'),
                   Output(component_id='totalScoreLvl1', component_property='children'),
                   Output(component_id='objslvl1', component_property='children'),

                   Output(component_id='squareAcc', component_property='children'),
                   Output(component_id='houseAcc', component_property='children'),
                   Output(component_id='smileyAcc', component_property='children'),
                   Output(component_id='squareSm', component_property='children'),
                   Output(component_id='houseSm', component_property='children'),
                   Output(component_id='smileySm', component_property='children'),
                   Output(component_id='totalScoreLvl4', component_property='children'),
                   Output(component_id='rainbow', component_property='figure'),
                   Output(component_id='rainbowsm', component_property='figure'),
                   Output(component_id='painting1', component_property='figure'),
                   Output(component_id='painting2', component_property='figure'),
                   Output(component_id='painting3', component_property='figure')

                   ],
                  ([Input('submit-button', 'n_clicks')],[Input('opt_dropdown','value')]),
                  [State('username', 'value'), State('controller', 'value')],
                  )
    def update_output(clicks, runChosen, username_value, controller_value):
        if (clicks is not None and username_value is not None):
            try:
                visualizer = Visualizer()
                opts, times = getRunIDs(visualizer, username_value)

                options = [{'label':f"Run ID {opts[i]},{times[i]}", 'value': opts[i]} for i in range(len(opts))]
                if runChosen[0] is not None:
                    visualizer.setArraysFromDB(visualizer.getDataFromDB(runChosen[0]))
                    r1,r2,r3,r4 = visualizer.getRangesDB(runChosen[0])
                else:
                    visualizer.setArraysFromDB(visualizer.getDataFromDB(opts[0]))
                    r1,r2,r3,r4 = visualizer.getRangesDB(opts[0])
                visualizer.initializeVectors(True)

                progressFig = getProgressFigs(opts)

                # Kitchen Tab
                olsm, ilsm, orism, irsm, olavg, ilavg, oriavg, iravg, avgSmoothnessApples, avgSpeedApples, totalscoreApples = visualizer.sparcOnApples()
                crsm2, cravg2, mssm2, msavg2, avgSmoothnessLVL2, avgSpeedLVL2, totalscoreLVL2 = visualizer.sparcOnLvl2()
                cksm1, ckavg1, cosm1, coavg1, avgSmoothnessLVL1, avgSpeedLVL1, totalscoreLVL1 = visualizer.sparcOnLvl1()

                fig3Dlvl3 = get3DFig(visualizer, 3)
                fig3Dlvl2 = get3DFig(visualizer, 2)
                fig3Dlvl1 = get3DFig(visualizer, 1)
                figSpeedlvl3 = getSpeedFig(visualizer, 3)
                figSpeedlvl2 = getSpeedFig(visualizer, 2)
                figSpeedlvl1 = getSpeedFig(visualizer, 1)

                # Toolshed Tab
                smsquare, smhouse, smsmiley,totalScoreLVL4 = visualizer.sparcOnLvl4()
                fig3Dlvl4 = get3DFig(visualizer, 4)
                fig3Dlvl5 = get3DFig(visualizer, 5)
                fig3Dlvl6 = get3DFig(visualizer, 6)
                fig3Drainbow = get3DFig(visualizer, 7)

                # Training Tab
                appleavg, coffeavg, drawingavg, appleTime, coffeetime, drawingtime = visualizer.getValues()

                # Objects Completed
                a,b,c= visualizer.getObjectives()
                objslvl3 = f"Missions Completed = {c}/4"
                objslvl2 = f"Missions Completed = {b}/2"
                objslvl1 = f"Missions Completed = {a}/2"

                # Painting Accuracies
                a,b,c = visualizer.getAccDB()


                e,f,g,h,i = visualizer.sparcOnRainbow()
                figRBProgress = getRBProgressFig(visualizer)

                return options, fig3Dlvl3, progressFig, figSpeedlvl3, olsm, ilsm, orism, irsm, avgSmoothnessApples, olavg, ilavg, oriavg, iravg, avgSpeedApples,\
                    f"Movement Quality Score = {totalscoreApples}",objslvl3, f"Right: {r1}",f"Left: {r2}",f"Up: {r3}",f"Forward: {r4}",\
                    crsm2, mssm2, avgSmoothnessLVL2, cravg2, msavg2, avgSpeedLVL2, fig3Dlvl2, figSpeedlvl2,f"Movement Quality Score = {totalscoreLVL2}",objslvl2, \
                    cosm1, cksm1, avgSmoothnessLVL1, coavg1, ckavg1, avgSpeedLVL1, fig3Dlvl1, figSpeedlvl1, f"Movement Quality Score = {totalscoreLVL1}",objslvl1, \
                    a,b,c,smsquare,smhouse,smsmiley, f"Movement Quality Score = {totalScoreLVL4}", fig3Drainbow, figRBProgress, fig3Dlvl4, fig3Dlvl5, fig3Dlvl6
            except:
                return emptyReturn()
        else:
            return emptyReturn()

    app.run_server(debug=False)