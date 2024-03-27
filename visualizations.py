import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

colorbewer = {1: ['#377eb8'],
              2: ['#377eb8','#e41a1c'],
              3: ['#377eb8','#e41a1c','#4daf4a'],
              4: ['#377eb8','#e41a1c','#4daf4a','#984ea3'],
              5: ['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00'],
              6: ['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33'],
              7: ['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628'],
              8: ['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf'],
              9: ['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
              10: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']}

def plot_topomap_comparison_highlight(proj_original, proj_new, 
                                      components_to_highligth, df_comp,
                                      hiertopomap=None):
    

    highligth = np.zeros(shape=proj_original.shape[0])

    for i, comp in enumerate(components_to_highligth):
        highligth[df_comp.loc[comp]['points']] = i+1
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'xy'},
                                {'type': 'xy'}]],
                        subplot_titles=('Original Projection - TopoMap',
                                        'New Projection - Hierarchical TopoMap'),
                        horizontal_spacing = 0.02)
    
    highligth_values = np.unique(highligth)
    l = len(np.unique(highligth))
    if not 0 in highligth_values:
        l += 1

    #colors = px.colors.qualitative.Plotly
    colors = colorbewer[l]

    for i in range(l):
        if i==0:
            if not 0 in highligth_values:
                continue
            name = 'Other points'
        else:
            name = f'Component {components_to_highligth[i-1]}'

        fig.add_trace(
            go.Scatter(x=proj_original[highligth==i,0], 
                    y=proj_original[highligth==i,1],
                    mode='markers',
                    #opacity=0.5,
                    marker=dict(
                        color=colors[i],
                        size=3,
                    ),
                    name=name,
                    legendgroup=name
                    ),
            row=1, col=1
        )

    if not hiertopomap is None:
        alphas = []
        for c in hiertopomap.components_to_scale:
            alphas.append(hiertopomap.components_info[c]['alpha'])

        #min_alpha = min(alphas)
        #range_alpha = max(alphas)-min_alpha
        max_alpha = max(alphas)
        max_color = sample_colorscale('Greys', [1])[0]

        for j in range(len(hiertopomap.components_info)):
            comp_ids = hiertopomap.components_info[j]['points']
            if 'hull' in hiertopomap.components_info[j].keys():
                hull = hiertopomap.components_info[j]['hull']
                points_ids = [comp_ids[i] for i in hull.vertices]
                points = list(hiertopomap.projections[points_ids,:])
                points.append(points[0])
                xs, ys = zip(*points)

                #alpha_scaled = (hiertopomap.components_info[j]['alpha']-min_alpha)/range_alpha
                alpha_scaled = (hiertopomap.components_info[j]['alpha']-1)/(1.1*max_alpha-1)
                hull_color = sample_colorscale('Greys', [alpha_scaled])[0]

                fig.add_trace(go.Scatter(x=xs, y=ys,
                                fill='toself', 
                                fillcolor = hull_color,
                                line_color=max_color,
                                opacity=0.5,
                                line_width=1,
                                text=f'Component {j}',
                                name='Components', legendgroup='Components',
                                showlegend=False,
                                ),
                            row=1, col=2)
                
    for i in range(l):
        if i==0:
            if not 0 in highligth_values:
                continue
            name = 'Other points'
        else:
            name = f'Component {components_to_highligth[i-1]}'

        fig.add_trace(
            go.Scatter(x=proj_new[highligth==i,0], 
                    y=proj_new[highligth==i,1],
                    mode='markers',
                    #opacity=0.5,
                    marker=dict(
                        color=colors[i],
                        size=3,
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=False
                    ),
            row=1, col=2
        )

    fig.update_layout(margin = dict(t=75, l=25, r=25, b=25),
            height=600,
            width=1200,
            legend= {'itemsizing': 'constant'},
            xaxis=dict(showticklabels=False), 
            yaxis=dict(showticklabels=False),
            xaxis2=dict(showticklabels=False), 
            yaxis2=dict(showticklabels=False)
            )

    return fig

def plot_hierarchical_treemap(df_comp, color='died_at'):
    fig = go.Figure(go.Treemap(
            labels=df_comp['id'],
            parents=df_comp['parent'],
            values=df_comp['size'],
            branchvalues='total',
            marker=dict(
                colors=df_comp[color],
                colorscale='Teal',
                showscale=True,
                colorbar=dict(
                    title='Persistence'
                )),
            hovertemplate='<b>Component #%{label} </b> <br> Points: %{value}<br> Persistence: %{color:.2f}<br> Parent: #%{parent}',
            name='',
            maxdepth=-1,
            )
        )

    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25),
                    title='TopoTree',
                    height=500,
                    width=800)
    
    return fig