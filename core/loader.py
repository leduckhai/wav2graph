import torch

from typing import Tuple, List
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from core.utils import Node

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            node_attributes: List[Node], 
            edges: List[Tuple[str, str]],
            embedding_id: str
        ):
        # Store the node attributes and edges
        self.node_attributes = node_attributes
        self.edges = edges

        # Create the node to index and index to node mappings
        self.node_to_index = {node.text: i for i, node in enumerate(node_attributes)}
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}

        # Store unique labels for the nodes
        self.labels = list(set([node.node_type for node in node_attributes]))

        # Create the data object
        if embedding_id == "random":
            self.tokenizer = None
            self.encoder = None
            self.data = self.create_data_with_random_features()
        else:
            # Load the transformer model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_id)
            self.encoder = AutoModel.from_pretrained(embedding_id).to(device)
            self.data = self.create_data()

    def create_data(self):
        # Create the edge index
        edge_index = torch.tensor([[self.node_to_index[edge[0]], self.node_to_index[edge[1]]] for edge in self.edges], dtype=torch.long).t()

        # Create the node features
        x = [self.tokenizer.encode(node.text) for node in self.node_attributes]
        x = [torch.tensor([e]) for e in x] # Add a dimension for batch
        x = torch.tensor([self.encoder(e).last_hidden_state.mean(dim=1)[0].tolist() for e in x])

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in self.node_attributes]
        y = torch.tensor(y, dtype=torch.long)
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels))
        return data
    
    def create_data_with_random_features(self):
        # Create the edge index
        edge_index = torch.tensor([[self.node_to_index[edge[0]], self.node_to_index[edge[1]]] for edge in self.edges], dtype=torch.long).t()

        # Create the node features
        x = torch.randn(len(self.node_attributes), 768).to(device)

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in self.node_attributes]
        y = torch.tensor(y, dtype=torch.long)

        # Calculate edge attributes by element-wise product of node embeddings with dim [num_edges, num_edge_features]
        edge_attr = x[edge_index[0]] * x[edge_index[1]]
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels), edge_attr=edge_attr)
        return data

    def __len__(self):
        return 1 # Only one graph in the dataset

    def __getitem__(self):
        return self.data

class MultiGraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            train_node_attributes: List[Node], 
            train_edges: List[Tuple[str, str]],
            test_node_attributes: List[Node], 
            test_edges: List[Tuple[str, str]],
            embedding_id: str
        ):
        # Store the node attributes and test_edges
        self.train_node_attributes = train_node_attributes
        self.train_edges = train_edges

        self.test_node_attributes = test_node_attributes
        self.test_edges = test_edges

        # Create the node to index and index to node mappings
        self.train_node_to_index = {node.text: i for i, node in enumerate(train_node_attributes)}
        self.train_index_to_node = {i: node for node, i in self.train_node_to_index.items()}

        self.test_node_to_index = {node.text: i for i, node in enumerate(test_node_attributes)}
        self.test_index_to_node = {i: node for node, i in self.test_node_to_index.items()}
        

        # Store unique labels for the nodes
        self.labels = list(set([node.node_type for node in train_node_attributes]))

        # Create the data object
        if embedding_id == "random":
            self.tokenizer = None
            self.encoder = None

            # Create the datasets
            self.train_dataset = self.create_data_with_random_features(self.train_node_to_index, self.train_node_attributes, self.train_edges)
            self.test_dataset = self.create_data_with_random_features(self.test_node_to_index, self.test_node_attributes, self.test_edges)
            self.data = self.merge_datasets(self.train_dataset, self.test_dataset)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_id)
            self.encoder = AutoModel.from_pretrained(embedding_id).to(device)

            # Create the datasets
            self.train_dataset = self.create_data(self.train_node_to_index, self.train_node_attributes, self.train_edges)
            self.test_dataset = self.create_data(self.test_node_to_index, self.test_node_attributes, self.test_edges)
            self.data = self.merge_datasets(self.train_dataset, self.test_dataset)

    def create_data(self, node_to_index, node_attributes, edges):
        # Create the edge index
        edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges], dtype=torch.long).t()

        # Create the node features
        x = [self.tokenizer.encode(node.text) for node in node_attributes]
        x = [torch.tensor([e]).to(device) for e in x] # Add a dimension for batch
        x = torch.tensor([self.encoder(e).last_hidden_state.mean(dim=1)[0].tolist() for e in x])

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in node_attributes]
        y = torch.tensor(y, dtype=torch.long)

        # Calculate edge attributes by element-wise product of node embeddings with dim [num_edges, num_edge_features]
        edge_attr = x[edge_index[0]] * x[edge_index[1]]
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels), edge_attr=edge_attr)
        return data
    
    def create_data_with_random_features(self, node_to_index, node_attributes, edges):
        # Create the edge index
        edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges], dtype=torch.long).t()

        # Create the node features with random values
        x = torch.randn(len(node_attributes), 768).to(device)

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in node_attributes]
        y = torch.tensor(y, dtype=torch.long)

        # Calculate edge attributes by element-wise product of node embeddings with dim [num_edges, num_edge_features]
        edge_attr = x[edge_index[0]] * x[edge_index[1]]
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels), edge_attr=edge_attr)
        return data

    def merge_datasets(self, train_dataset, test_dataset):
        # Concatenate node features and labels
        combined_x = torch.cat([train_dataset.x, test_dataset.x], dim=0)
        combined_y = torch.cat([train_dataset.y, test_dataset.y], dim=0)
        
        # Adjust edge indices for test dataset
        num_train_nodes = train_dataset.x.size(0)
        adjusted_test_edge_index = test_dataset.edge_index + num_train_nodes
    
        # Concatenate edge indices and edge attributes
        combined_edge_index = torch.cat([train_dataset.edge_index, adjusted_test_edge_index], dim=1)
        combined_edge_attr = torch.cat([train_dataset.edge_attr, test_dataset.edge_attr], dim=0)
        
        # Create train and test masks
        train_mask = torch.zeros(combined_x.size(0), dtype=torch.bool)
        test_mask = torch.zeros(combined_x.size(0), dtype=torch.bool)
        
        train_mask[:num_train_nodes] = True
        test_mask[num_train_nodes:] = True
        
        # Create the combined Data object
        combined_data = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            edge_attr=combined_edge_attr,
            y=combined_y,
            num_classes=train_dataset.num_classes,  # Assuming both datasets have the same number of classes
            train_mask=train_mask,
            test_mask=test_mask
        )
        
        return combined_data

    def __len__(self):
        return 1 # Only one graph in the dataset

    def __getitem__(self):
        return self.data
