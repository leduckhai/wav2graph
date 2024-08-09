import json
import yaml
import torch

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from pydantic import BaseModel, Field
from typing import List
from core.model import *


class Node(BaseModel):
    text: str
    node_type: str
    features: List[str] = Field(default=[], description="List of features for the node.")

# Create a class to switch between models
class ModelNodeSwitcher:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, in_channels, out_channels):
        if self.model_name == "SAGE":
            return GraphSAGE(in_channels, out_channels)
        elif self.model_name == "GCN":
            return GCN(in_channels, out_channels)
        elif self.model_name == "GAT":
            return GAT(in_channels, out_channels)
        elif self.model_name == "SuperGAT":
            return SuperGAT(in_channels, out_channels)
        else:
            raise ValueError("Model not found")
        
# Create class to switch between models
class ModelLinkSwitcher:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, in_channels, out_channels):
        if self.model_name == 'SAGE':
            return GraphSAGEEncoder(in_channels, out_channels)
        elif self.model_name == 'GCN':
            return GCNEncoder(in_channels, out_channels)
        elif self.model_name == 'GAT':
            return GATEncoder(in_channels, out_channels)
        elif self.model_name == 'SuperGAT':
            return SuperGATEncoder(in_channels, out_channels)
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')

def read_config(file_path):
    """Read and parse a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data.

    Raises:
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def get_nodes_and_edges(json_file):
    """Function to get nodes and edges from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    _, kg_edges, node_attributes = data
    kg_nodes = [
        Node(**node, features=node.get("entity_type", [])) 
        for node in node_attributes
    ]

    return kg_nodes, kg_edges

def convert_to_link_prediction_data(data):
    # Number of nodes and edges
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    
    # Existing edges (positive edges)
    pos_edge_index = data.edge_index
    pos_edge_label = torch.ones(pos_edge_index.size(1), dtype=torch.float)
    
    # Sample negative edges (non-existing edges)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, num_nodes=num_nodes, num_neg_samples=pos_edge_index.size(1)
    )
    neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float)
    
    # Combine positive and negative edge labels and their indices
    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    
    # Create new Data object with the required attributes
    link_pred_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        y=data.y,
        num_classes=data.num_classes,
        pos_edge_label=pos_edge_label,
        pos_edge_label_index=pos_edge_index,
        neg_edge_label=neg_edge_label,
        neg_edge_label_index=neg_edge_index
    )
    
    return link_pred_data
