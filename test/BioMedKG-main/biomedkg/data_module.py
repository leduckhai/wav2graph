from lightning import LightningDataModule
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler
from typing import Callable

from biomedkg.modules.data import PrimeKG, BioKG
from biomedkg.configs import data_settings, train_settings

class PrimeKGModule(LightningDataModule):
    def __init__(
            self, 
            data_dir : str = data_settings.DATA_DIR, 
            process_node_lst : set[str] = data_settings.NODES_LST,
            process_edge_lst : set[str] = data_settings.EDGES_LST,
            batch_size : int = train_settings.BATCH_SIZE,
            val_ratio : float = train_settings.VAL_RATIO,
            test_ratio : float = train_settings.TEST_RATIO,
            encoder : Callable = None
            ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.process_node_lst = process_node_lst
        self.process_edge_lst = process_edge_lst
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.encoder = encoder

    def setup(self, stage : str = "split", embed_dim: int = None):
        self.primekg = PrimeKG(
            data_dir=self.data_dir,
            process_node_lst=self.process_node_lst,
            process_edge_lst=self.process_edge_lst,
            embed_dim=embed_dim,
            encoder=self.encoder
        )
        self.edge_map_index = self.primekg.edge_map_index

        self.data = self.primekg.get_data()

        if stage == "split":
            self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
                num_val=self.val_ratio,
                num_test=self.test_ratio,
                neg_sampling_ratio=0.,
            )(data=self.data)
    
    def subgraph_dataloader(self,):
        return NeighborLoader(
            data=self.data,
            num_neighbors=[-1],
            num_workers=0,
            shuffle=False,
        )
       
    def all_dataloader(self):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def train_dataloader(self, loader_type: str = "neighbor"):
        
        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.train_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
                shuffle=True,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.train_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=1000,
                num_workers=0,
            )

    def val_dataloader(self, loader_type: str):

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.val_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.val_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=200,
                num_workers=0,
            )

    def test_dataloader(self, loader_type: str):

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.test_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.test_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=200,
                num_workers=0,
            )
        
class BioKGModule(LightningDataModule):
    def __init__(
            self, 
            data_dir : str = data_settings.BENCHMARK_DIR,
            batch_size : int = train_settings.BATCH_SIZE,
            val_ratio : float = train_settings.VAL_RATIO,
            test_ratio : float = train_settings.TEST_RATIO,
            encoder: Callable = None,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.encoder = encoder

    def setup(self, stage: str = "split", embed_dim: int = None):
        self.biokg = BioKG(
            data_dir=self.data_dir,
            embed_dim=embed_dim,
            encoder=self.encoder
        )
        self.edge_map_index = self.biokg.edge_map_index
        self.data = self.biokg.get_data()

        if stage == "split":
            self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
                num_val=self.val_ratio,
                num_test=self.test_ratio,
                neg_sampling_ratio=0.,
            )(data=self.data)
            
    def subgraph_dataloader(self):
        return NeighborLoader(
            data=self.data,
            num_neighbors=[-1],
            num_workers=0,
            shuffle=False,
        )
    
    def all_dataloader(self):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def train_dataloader(self, loader_type: str = "neighbor"):
        
        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.train_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
                shuffle=True,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.train_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=1000,
                num_workers=0,
            )

    def val_dataloader(self, loader_type: str):

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.val_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.val_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=200,
                num_workers=0,
            )

    def test_dataloader(self, loader_type: str):

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.test_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "graph_saint":
            return GraphSAINTRandomWalkSampler(
                data=self.test_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=200,
                num_workers=0,
            )