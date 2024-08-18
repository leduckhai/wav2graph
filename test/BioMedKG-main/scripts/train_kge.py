import os
import time
import json
import argparse

import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.kge_module import KGEModule
from biomedkg.factory import NodeEncoderFactory
from biomedkg.data_module import PrimeKGModule, BioKGModule
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings, kge_settings, data_settings, node_settings

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--task', 
            type=str, 
            action='store', 
            choices=['train', 'test'], 
            default='train', 
            help="Do training or testing task")
    
    parser.add_argument(
            '--gcl_embed_path',
            type=str,
            default=None,
            required=False,
            help="Path to your GCL embedding")

    parser.add_argument(
            '--run_benchmark',
            action="store_true",
            help="Run the benchmark")
    
    parser.add_argument(
            '--node_init_method',
            type=str, 
            action='store', 
            required=False,
            default="gcl",
            choices=["random", "gcl", "llm"],
            help="Select node init method")
    
    parser.add_argument(
            '--modality_transform_method', 
            type=str, 
            action='store', 
            required=False,
            default='None',
            choices=['attention', 'redaf', 'None'], 
            help="Modality transform methods")
    
    parser.add_argument(
            '--ckpt_path', 
            type=str, 
            default=None,
            required=False,
            help="Path to checkpoint")
    
    opt = parser.parse_args()
    return opt

def main(
        task:str, 
        ckpt_path: str, 
        gcl_embed_path: str,
        run_benchmark: bool,
        node_init_method: str,
        modality_transform_method: str
        ):
    seed_everything(train_settings.SEED)

    if node_init_method in ["random", "gcl"]:
        modality_transform_method = None
    
    if gcl_embed_path is not None:
        node_init_method = "gcl"

    node_encoder, embed_dim = NodeEncoderFactory.create_encoder(
        node_init_method=node_init_method,
        gcl_embed_path=gcl_embed_path,
    )

    # Setup data module
    if run_benchmark:
        assert ckpt_path is not None

        data_module = BioKGModule(encoder=node_encoder)
        data_module.setup(stage="split", embed_dim=embed_dim)

        model = KGEModule.load_from_checkpoint(ckpt_path)
        # In PrimeKG, drug - gene relation is one
        model.select_edge_type_id = 1

    else:
        data_module = PrimeKGModule(encoder=node_encoder)
        data_module.setup(stage="split", embed_dim=embed_dim)
        model = KGEModule(
            in_dim=embed_dim,
            num_relation=data_module.data.num_edge_types,
            node_init_method=node_init_method,
            modality_transform_method=modality_transform_method
        )

    model.edge_mapping = data_module.edge_map_index

    # Setup logging/ckpt path
    if ckpt_path is None:
        log_id = str(int(time.time()))
        if gcl_embed_path is not None:
            model_name = gcl_embed_path.split('/')[-1]
            exp_name = f"{node_init_method}_{model_name}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{log_id}"
            ckpt_dir = os.path.join(train_settings.OUT_DIR, "kge", exp_name)
            log_dir = os.path.join(train_settings.LOG_DIR, "kge", exp_name)
        else:   
            exp_name = f"{node_init_method}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{log_id}"
            ckpt_dir = os.path.join(train_settings.OUT_DIR, "kge", exp_name)
            log_dir = os.path.join(train_settings.LOG_DIR, "kge", exp_name)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        if run_benchmark:
            ckpt_dir = os.path.dirname(ckpt_path) + "_benchmark"
            exp_name = "benchmark" + str(os.path.basename(ckpt_dir))
        else:
            ckpt_dir = os.path.dirname(ckpt_path)
            exp_name = str(os.path.basename(ckpt_dir).split("_")[-1])

        log_dir = ckpt_dir.replace(os.path.basename(train_settings.OUT_DIR), os.path.basename(train_settings.LOG_DIR))

    # Prepare trainer args
    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 10,
        "deterministic": True, 
    }

    if torch.cuda.device_count() > 1:
        trainer_args.update(
            {
                "devices": train_settings.DEVICES,
            }
        )

    # Train
    if task == "train":
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            monitor="val_loss", 
            save_top_k=3, 
            mode="min",
            save_last=True,
            )
        
        early_stopping = EarlyStopping(monitor="val_AUROC_mean", mode="max", min_delta=0.001, patience=1)

        logger = CometLogger(
            api_key=find_comet_api_key(),
            project_name="BioMedKG-KGE",
            save_dir=log_dir,
            experiment_name=exp_name,
        )

        trainer_args.update(
            {
                "max_epochs": train_settings.EPOCHS,
                "check_val_every_n_epoch": train_settings.VAL_EVERY_N_EPOCH,
                "enable_checkpointing": True,     
                "gradient_clip_val": 1.0,
                "callbacks": [checkpoint_callback, early_stopping],
                "default_root_dir": ckpt_dir,
                "logger": logger, 
            }
        )

        trainer = Trainer(**trainer_args)

        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(loader_type="graph_saint"),
            val_dataloaders=data_module.val_dataloader(loader_type="graph_saint"),
            ckpt_path=ckpt_path 
        )

    # Test
    elif task == "test":
        assert ckpt_path is not None, "Please specify checkpoint path."
        trainer = Trainer(**trainer_args)
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="graph_saint"),
             ckpt_path=ckpt_path,
        )
    
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main(**vars(parse_opt()))