import argparse
import numpy as np
import pandas as pd 
from scipy import stats 
from sklearn.model_selection import KFold
from lightning.pytorch import Trainer, seed_everything
from torch_geometric.loader import LinkNeighborLoader
from biomedkg.modules.data import TripletBase
from biomedkg.modules.node import EncodeNode
from biomedkg.configs import node_settings, train_settings, data_settings
from biomedkg.kge_module import KGEModule

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--ckpt_path', 
            type=str, 
            default=None,
            required=False,
            help="Path to checkpoint")
    
    parser.add_argument(
            '--gcl_embed_path',
            type=str,
            default=None,
            required=False,
            help="Path to your GCL embedding")

    parser.add_argument(
            '--max_epoch',
            type=int,
            default=10,
            help="Max epochs")

    parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help="Set seed")
    
    parser.add_argument(
            '--neg_ratio',
            type=int,
            default=10,
            help="Set seed")
    
    opt = parser.parse_args()
    return opt


def run_fold(model, train_graph, test_graph, max_epoch):

    train_graph_loader = LinkNeighborLoader(
            data=train_graph,
            batch_size=train_settings.BATCH_SIZE,
            num_neighbors=[30] * 3,
            num_workers=4,
        )

    test_graph_loader = LinkNeighborLoader(
        data=test_graph,
        batch_size=train_settings.BATCH_SIZE,
        num_neighbors=[30] * 3,
        num_workers=4,
    )
    
    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 2,
        "deterministic": True, 
        "max_epochs": max_epoch,
        "gradient_clip_val": 1.0,
    }

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=model,
        train_dataloaders=train_graph_loader,
    )

    return trainer.test(dataloaders=test_graph_loader)[0]


def summarize(result: list[dict]):
    result_df = pd.DataFrame(result)
    print("\n---------------------------\n")
    for col in result_df:
        metrics_mean = np.mean(result_df[col].values)
        metrics_std = np.std(result_df[col].values)
        t_stat, p_value = stats.ttest_1samp(result_df[col].values, 0)
        print(f"{col}: {round(metrics_mean, 4)} ± {round(metrics_std, 4)}, p-value: {p_value:.4f}")
    print("\n---------------------------\n")


def k_fold(ckpt_path:str, gcl_embed_path: str, max_epoch:int, seed: int, neg_ratio: int):
    df = pd.read_csv(data_settings.BENCHMARK_DIR)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    kfold.get_n_splits(df)

    encoder = EncodeNode(embed_path=gcl_embed_path)
    embed_dim = node_settings.GCL_TRAINED_NODE_DIM

    seed_everything(seed)
    result = list()

    for train_index, test_index in kfold.split(df):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        train_triplet = TripletBase(df=train_data, embed_dim=embed_dim, encoder=encoder)
        train_graph = train_triplet.get_data()

        test_triplet = TripletBase(df=test_data, embed_dim=embed_dim, encoder=encoder).get_data()
        test_graph = test_triplet.get_data()

        model = KGEModule.load_from_checkpoint(ckpt_path)
        model.neg_ratio = neg_ratio
        model.select_edge_type_id = 1
        model.edge_mapping = train_triplet.edge_map_index

        fold_result = run_fold(
            model=model,
            train_graph=train_graph,
            test_graph=test_graph,
            max_epoch=max_epoch,
        )

        result.append(fold_result)

    summarize(result)


if __name__ == "__main__":
    k_fold(**vars(parse_opt()))