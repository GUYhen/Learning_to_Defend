import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import pickle
import os

class NetworkTopology:
    def __init__(self, num_nodes: int, critical_nodes: List[int], 
                 connection_prob: float = 0.15, min_degree: int = 2, max_degree: int = 8):
        self.num_nodes = num_nodes
        self.critical_nodes = critical_nodes
        self.connection_prob = connection_prob
        self.min_degree = min_degree
        self.max_degree = max_degree
        
        self.adjacency_matrix = None
        self.degrees = None
        self.node_weights = None
        self.graph = None
        self.topology_fixed = False
        
    def generate_fixed_topology(self, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        
        while True:
            G = nx.erdos_renyi_graph(self.num_nodes, self.connection_prob, seed=seed)
            
            degrees = dict(G.degree())
            
            for critical_node in self.critical_nodes:
                current_degree = degrees[critical_node]
                target_degree = max(6, current_degree)
                
                while degrees[critical_node] < target_degree:
                    candidates = [i for i in range(self.num_nodes) 
                                if i != critical_node and not G.has_edge(critical_node, i)
                                and degrees[i] < self.max_degree]
                    
                    if not candidates:
                        break
                        
                    target = np.random.choice(candidates)
                    G.add_edge(critical_node, target)
                    degrees[critical_node] += 1
                    degrees[target] += 1
            
            if nx.is_connected(G):
                final_degrees = dict(G.degree())
                if (all(self.min_degree <= final_degrees[i] <= self.max_degree 
                       for i in range(self.num_nodes)) and
                    all(final_degrees[critical] >= 6 for critical in self.critical_nodes)):
                    break
            
            seed += 1
        
        self.graph = G
        self.adjacency_matrix = nx.adjacency_matrix(G).toarray()
        self.degrees = np.array([final_degrees[i] for i in range(self.num_nodes)])
        self.node_weights = self.degrees / np.sum(self.degrees)
        self.topology_fixed = True
        
        self.save_topology()
        
        return self.adjacency_matrix
    
    def load_or_generate_topology(self, topology_file: str = "network_topology.pkl") -> np.ndarray:
        if os.path.exists(topology_file):
            self.load_topology(topology_file)
        else:
            self.generate_fixed_topology()
        return self.adjacency_matrix
    
    def save_topology(self, filepath: str = "network_topology.pkl"):
        topology_data = {
            'adjacency_matrix': self.adjacency_matrix,
            'degrees': self.degrees,
            'node_weights': self.node_weights,
            'critical_nodes': self.critical_nodes,
            'num_nodes': self.num_nodes
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(topology_data, f)
    
    def load_topology(self, filepath: str = "network_topology.pkl"):
        with open(filepath, 'rb') as f:
            topology_data = pickle.load(f)
        
        self.adjacency_matrix = topology_data['adjacency_matrix']
        self.degrees = topology_data['degrees']
        self.node_weights = topology_data['node_weights']
        self.critical_nodes = topology_data['critical_nodes']
        self.num_nodes = topology_data['num_nodes']
        self.topology_fixed = True
        
        self.graph = nx.from_numpy_array(self.adjacency_matrix)
    
    def get_node_neighbors(self, node_id: int) -> List[int]:
        return list(np.where(self.adjacency_matrix[node_id] == 1)[0])
    
    def is_critical_node(self, node_id: int) -> bool:
        return node_id in self.critical_nodes
    
    def get_network_connectivity(self) -> float:
        if self.graph is None:
            return 0.0
        return nx.edge_connectivity(self.graph) / self.num_nodes
    
    def visualize_topology(self, save_path: str = None, node_states: Optional[np.ndarray] = None):
        if self.graph is None:
            return
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.graph, seed=42)
        
        node_colors = []
        for i in range(self.num_nodes):
            if node_states is not None and node_states[i] == 0:
                node_colors.append('gray')
            elif i in self.critical_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        node_sizes = [100 + 50 * self.degrees[i] for i in range(self.num_nodes)]
        
        nx.draw(self.graph, pos, node_color=node_colors, node_size=node_sizes,
                with_labels=True, font_size=8, font_weight='bold')
        
        plt.title("Network Topology (Red: Critical Nodes, Gray: Compromised)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_topology_info(self) -> Dict:
        return {
            'num_nodes': self.num_nodes,
            'num_edges': np.sum(self.adjacency_matrix) // 2,
            'critical_nodes': self.critical_nodes,
            'degrees': self.degrees.tolist(),
            'average_degree': float(np.mean(self.degrees)),
            'max_degree': int(np.max(self.degrees)),
            'min_degree': int(np.min(self.degrees)),
            'connectivity': self.get_network_connectivity()
        } 