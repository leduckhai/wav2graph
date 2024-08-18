import argparse
import torch
import wandb
import os, sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from huggingface_hub import login
from torch.optim import Adam
from torch_geometric.nn import VGAE
from core.metrics import auc_score, ap_score
from core.utils import ModelLinkSwitcher, read_config, get_nodes_and_edges, convert_to_link_prediction_data
from core.loader import MultiGraphDataset

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

def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse arguments
    parser = argparse.ArgumentParser(description='Node Inductive Learning')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--embedding', type=str, required=True, help='Name of the embedding model')
    parser.add_argument('--type', type=str, default='Transductive', help='Type of the task')
    args = parser.parse_args()

    # Read the config file
    config = read_config(args.config)

    # Login to the HuggingFace Hub
    login(config["hf_token"])

    # Load the dataset
    train_nodes, train_edges = get_nodes_and_edges(config["train_data"])
    test_nodes, test_edges = get_nodes_and_edges(config["test_data"])

    embedding_id = args.embedding
    dataset = MultiGraphDataset(train_nodes, train_edges, test_nodes, test_edges, embedding_id)
    # dataset = MultiGraphDataset(train_nodes[:400], train_edges[:200], test_nodes[:400], test_edges[:200], embedding_id)
    data = dataset.data.to(device)

    # Merge datasets
    train_data = convert_to_link_prediction_data(dataset.train_dataset)
    test_data = convert_to_link_prediction_data(dataset.test_dataset)

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
            test_auc, test_ap = test(model, test_data)

            wandb.log({
                'loss': loss,
                'train_auc': train_auc,
                'train_ap': train_ap,
                'test_auc': test_auc,
                'test_ap': test_ap
            })
            
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch:03d}, Loss: {loss:.4f}, \
                    AUC Train: {train_auc:.4f}, AP Train: {train_ap:.4f}, \
                    AUC Test: {test_auc:.4f}, AP Test: {test_ap:.4f}'
                )

        # Log model name
        wandb.log({"Model": model_name})
        wandb.log({"Type": args.type})

        # Log additional information
        wandb.log({"Num epochs": num_epochs})
        wandb.log({"Optimizer": optimizer.__class__.__name__})
        wandb.log({"Learning rate": learning_rate})
        wandb.log({"Embedding": embedding_id})
        wandb.log({"Out channels": config['train']['out_channels']})

        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
