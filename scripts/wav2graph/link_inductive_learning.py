import argparse
import torch
import wandb
import os, sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from huggingface_hub import login
from torch_geometric.transforms import RandomLinkSplit
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.nn import VGAE
from core.metrics import auc_score, ap_score
from core.utils import ModelLinkSwitcher, read_config, get_nodes_and_edges
from core.loader import GraphDataset

# Create the trainer
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index)
    loss.backward()
    optimizer.step()

    # Calculate the auc, ap
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return loss, auc, ap

def validate(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap

def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse arguments
    parser = argparse.ArgumentParser(description='Link Inductive Learning')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--embedding', type=str, required=True, help='Name of the embedding model')
    args = parser.parse_args()

    # Read the config file
    config = read_config(args.config)

    # Login to the HuggingFace Hub
    login(config["hf_token"])

    # Load the dataset
    nodes, edges = get_nodes_and_edges(config["data"])

    embedding_id = args.embedding
    dataset = GraphDataset(nodes, edges, embedding_id)
    # dataset = GraphDataset(nodes[:400], edges[:200], embedding_id)

    transform = RandomLinkSplit(**config["splitted_edge_kwargs"], edge_types=None)
    train_data, val_data, test_data = transform(dataset.data)

    # Create the model
    model_names = config["train"]["model_names"]
    learning_rate = config["train"]["learning_rate"]
    num_epochs = config["train"]["num_epochs"]
    for model_name in model_names:
        model = VGAE(ModelLinkSwitcher(model_name)(train_data.num_features, config['train']['out_channels'])).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=config['train']['weight_decay'])

        # Log in wandb
        wandb.init(project=config["train"]["project_name"], tags=['link-prediction'])

        for epoch in range(1, num_epochs + 1):
            loss, train_auc, train_ap = train(model, train_data, optimizer)
            val_auc, val_ap = validate(model, val_data)
            test_auc, test_ap = test(model, test_data)

            wandb.log({
                'loss': loss,
                'train_auc': train_auc,
                'train_ap': train_ap,
                'val_auc': val_auc,
                'val_ap': val_ap,
                'test_auc': test_auc,
                'test_ap': test_ap
            })
            
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch:03d}, Loss: {loss:.4f}, \
                    AUC Train: {train_auc:.4f}, AP Train: {train_ap:.4f}, \
                    AUC Val: {val_auc:.4f}, AP Val: {val_ap:.4f}, \
                    AUC Test: {test_auc:.4f}, AP Test: {test_ap:.4f}'
                )

        # Log model name
        wandb.log({"Model": model_name})
        wandb.log({"Type": "Inductive"})

        # Log additional information
        wandb.log({"Num epochs": num_epochs})
        wandb.log({"Optimizer": optimizer.__class__.__name__})
        wandb.log({"Learning rate": learning_rate})
        wandb.log({"Embedding": embedding_id})
        wandb.log({"Out channels": config['train']['out_channels']})
        wandb.log({"split": config['splitted_edge_kwargs']})

        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
