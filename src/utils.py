
import time
import yaml
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper



def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def add_weights_to_nx_graph(nx_graph):
    for u, v in nx_graph.edges():
        w = round(random.uniform(0, 1), 2)
        while w == 0:
            w = round(random.uniform(0, 1), 2)
        nx_graph[u][v]["weight"] = w
    return nx_graph


def generate_er_graphs(n_graphs, n_nodes):

    graphs = {}

    for i in range(n_graphs):

        p = random.randrange(6, 9) / 10

        g = nx.erdos_renyi_graph(
            n=n_nodes,
            p=p
        )

        g = add_weights_to_nx_graph(g)

        graphs[f"er_graph_{i}"] = g

    return graphs
