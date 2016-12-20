import words
import json
import os
import csv

import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *

def rank_density(input_path="", output_path=""):
    if input_path and output_path:
        input_folder = input_path
        output_folder = output_path

    elif input_path and not output_path:
        input_folder = input_path
        output_folder = os.path.join("data/ranked_out", os.path.basename(input_folder))
    else:
        input_folder = "data/semgraphs"
        output_folder = "data/ranked_out"

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.startswith("."):
                with open(os.path.join(root, file), "rU") as input:
                    semgraph = json.load(input)
                    top_n = words.top_n_words(semgraph, 50)
                    just_words = []
                    for entry in top_n:
                        just_words.append((entry[0], len(entry[1])))

                    if not input_path:
                        final_out_folder = root.replace(input_folder, output_folder)
                    else:
                        final_out_folder = os.path.join(output_folder, os.path.basename(root))

                    if not os.path.isdir(final_out_folder):
                        os.makedirs(final_out_folder)

                    final_out = os.path.join(final_out_folder, file+".csv")
                    with open(final_out, "wb") as output:
                        writer = csv.writer(output)
                        for word in just_words:
                            writer.writerow([word[0], word[1]])

def plot_semantic_graph(sem_graph):
    with open(sem_graph.path, "rU") as input:
        semgraph = json.load(input)

    N = len(semgraph)
    G = nx.MultiGraph()

    for key in semgraph:
        G.add_node(key, word=key, size=1)

    for key, value in semgraph.items():
        if value:
            for element in value:
                G.add_edge(key, element[0])

    pos = nx.fruchterman_reingold_layout(G)
    nx.set_node_attributes(G, 'pos', pos)

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='Portland',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='# Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        the_node = G.node[node]
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['marker']['color'].append(len(G.neighbors(node)))
        node_trace['text'].append('{}<br># connections: {}'.format(the_node['word'], len(G.neighbors(node))))

    fig_text = "{} {}() > {}".format(sem_graph.source, sem_graph.sim_func, 1 - sem_graph.threshold)
    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                     title='<br>Semantic Graph',
                     titlefont=dict(size=16),
                     showlegend=False,
                     width=650,
                     height=650,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text=fig_text,
                         # Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    return fig