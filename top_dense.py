import words
import json
import os
import csv
import re

import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from scipy import stats

from words import *

cos_regx = re.compile('(cosine_0)(\\.)(\\d+)')

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
    G = nx.Graph()

    for key in semgraph:
        G.add_node(key, word=key, size=1)

    for key, value in semgraph.items():
        if value:
            for element in value:
                if element[0] not in G.neighbors(key):
                    G.add_edge(key, element[0])

    pos = nx.fruchterman_reingold_layout(G)
    nx.set_node_attributes(G, 'pos', pos)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.3, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
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
                title='# Edges',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        # print "{} : {}".format(node, G.neighbors(node))
        the_node = G.node[node]
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['marker']['color'].append(len(G.neighbors(node)))
        node_trace['text'].append('{}<br># edges: {}'.format(the_node['word'], len(G.neighbors(node))))

    fig_text = u"{} {}({}) {} {}".format(sem_graph.source, sem_graph.sim_func,
                                         u"\u03B8",  u"\u2265",  1 - sem_graph.threshold)
    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                 layout=go.Layout(
                     title=u'<br>{}'.format(fig_text),
                     titlefont=dict(size=16),
                     showlegend=False,
                     width=650,
                     height=650,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text="",
                         # Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    return fig


def plot_edges_vs_month_compr(data, month):
    trace = go.Scatter(
        x=data['edges'],
        y=data[month],
        text=data['word'],
        mode='markers'
    )

    layout = go.Layout(
        title='Semantic Density vs. Month {} Comprehension Score'.format(month),
        hovermode='closest',
        xaxis=dict(
            title='Edges',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='WordBank Comprehension Score',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=False
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='basic-scatter')


def top_correlation_threshold(path, wb, wb_month, source):
    thresh = 0
    top_graph = None
    top_corr = 0
    top_p = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("."):
                cos_result = cos_regx.search(root)
                if cos_result:
                    thresh = float("0." + cos_result.group(3))

                graph = SemanticGraph(source=source, sim_func="cos",
                                      thresh=thresh, path=os.path.join(root, file),
                                      wb=wb)
                top_n = graph.top_n_dense(all=True)
                corr_coef, p = stats.pearsonr(top_n['edges'], top_n[wb_month])
                if corr_coef > top_corr:
                    top_corr = corr_coef
                    top_p = p
                    top_graph = graph

    return top_graph, top_corr, top_p




