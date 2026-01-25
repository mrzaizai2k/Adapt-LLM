import sys
sys.path.append("")

import numpy as np
import networkx as nx
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import hashlib
from tqdm.auto import tqdm
from typing import List, Dict, Optional

from src.embedding.embedding_utils import Estimator

    
class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        use_node_attribute (Optional[str]): Optional attribute name to be used.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        use_node_attribute: Optional[str],
        erase_base_features: bool,
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.use_node_attribute = use_node_attribute
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        if self.use_node_attribute is not None:
            # We retrieve the features of the nodes with the attribute name
            # `feature` and assign them into a dictionary with structure:
            # {node_a_name: feature_of_node_a}
            # Nodes without this feature will not appear in the dictionary.
            features = nx.get_node_attributes(self.graph, self.use_node_attribute)

            # We check whether all nodes have the requested feature
            if len(features) != self.graph.number_of_nodes():
                missing_nodes = []
                # We find up to five missing nodes so to make
                # a more informative error message.
                for node in tqdm(
                    self.graph.nodes,
                    total=self.graph.number_of_nodes(),
                    leave=False,
                    dynamic_ncols=True,
                    desc="Searching for missing nodes"
                ):
                    if node not in features:
                        missing_nodes.append(node)
                    if len(missing_nodes) > 5:
                        break
                raise ValueError(
                    (
                        "We expected for ALL graph nodes to have a node "
                        "attribute name `{}` to be used as part of "
                        "the requested embedding algorithm, but only {} "
                        "out of {} nodes has the correct attribute. "
                        "Consider checking for typos and missing values, "
                        "and use some imputation technique as necessary. "
                        "Some of the nodes without the requested attribute "
                        "are: {}"
                    ).format(
                        self.use_node_attribute,
                        len(features),
                        self.graph.number_of_nodes(),
                        missing_nodes
                    )
                )
            # If so, we assign the feature set.
            self.features = features
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)]
                                   for k, v in self.features.items()}

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + \
                sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]
    
class Graph2Vec(Estimator):
    r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurrence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 500,
        workers: int = 1,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
        ):

            self.wl_iterations = wl_iterations
            self.attributed = attributed
            self.dimensions = dimensions
            self.workers = workers
            self.down_sampling = down_sampling
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.min_count = min_count
            self.seed = seed
            self.erase_base_features = erase_base_features

    def fit(self, graphs: List[nx.classes.graph.Graph]):
            """
            Fitting a Graph2Vec model.

            Arg types:
                * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
            """
            self._set_seed()
            graphs = self._check_graphs(graphs)
            
            documents = [
                WeisfeilerLehmanHashing(
                    graph,
                    self.wl_iterations,
                    "feature" if self.attributed else None,
                    self.erase_base_features,
                )
                for graph in graphs
            ]

            documents = [
                TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
                for i, doc in enumerate(documents)
            ]

            self.model = Doc2Vec(
                documents,
                vector_size=self.dimensions,
                window=0,
                min_count=self.min_count,
                dm=0,
                sample=self.down_sampling,
                workers=self.workers,
                epochs=self.epochs,
                alpha=self.learning_rate,
                seed=self.seed,
            )

            self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents)]


    def get_embedding(self) -> np.array:
            r"""Getting the embedding of graphs.

            Return types:
                * **embedding** *(Numpy array)* - The embedding of graphs.
            """
            return np.array(self._embedding)


    def infer(self, graphs) -> np.array:
            """Infer the graph embeddings.
        
            Arg types:
                * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

            Return types:
                * **embedding** *(Numpy array)* - The embedding of graphs.
            """
            self._set_seed()
            graphs = self._check_graphs(graphs)
            
            documents = [
                WeisfeilerLehmanHashing(
                    graph,
                    self.wl_iterations,
                    "feature" if self.attributed else None,
                    self.erase_base_features,
                )
                for graph in graphs
            ]


            documents = [doc.get_graph_features() for _, doc in enumerate(documents)]

            embedding = np.array(
                [
                    self.model.infer_vector(
                        doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
                    )
                    for doc in documents
                ]
            )

            return embedding

if __name__ == "__main__":
    import networkx as nx
    import numpy as np

    # ----- Create 3 sample graphs -----
    # Graph 1: triangle
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Graph 2: line
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Graph 3: star
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])

    graphs = [G1, G2, G3]

    for G in graphs:
        G.add_edges_from((i, i) for i in G.nodes())

    # ----- Initialize Graph2Vec -----
    model = Graph2Vec(
        wl_iterations=2,
        attributed=False,
        dimensions=500,     # 👈 500-dimensional embedding
        workers=1,
        epochs=20,
        min_count=1,
        seed=42
    )

    # ----- Fit model -----
    model.fit(graphs)

    # ----- Get embeddings -----
    embeddings = model.get_embedding()
    embeddings = np.round(embeddings, 2)


    print("Embedding shape:", embeddings.shape)
    print("Embeddings:")
    print(embeddings)
