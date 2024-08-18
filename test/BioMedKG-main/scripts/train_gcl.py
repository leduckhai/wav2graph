import os
import time
import json
import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.gcl_module import DGIModule, GRACEModule, GGDModule
from biomedkg.data_module import PrimeKGModule
from biomedkg.factory import NodeEncoderFactory
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings, gcl_settings, node_settings


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
            '--model_name', 
            type=str, 
            action='store', 
            choices=['dgi', 'grace', 'ggd'], 
            default='dgi', 
            help="Select contrastive model name")
    
    parser.add_argument(
            '--node_type', 
            type=str, 
            action='store', 
            required=True,
            choices=['gene', 'drug', 'disease'], 
            help="Train contrastive learning on which node type")   

    parser.add_argument(
            '--node_init_method',
            type=str, 
            action='store', 
            required=True,
            choices=["random", "llm"],
            help="Select node init method")    
    
    parser.add_argument(
            '--modality_transform_method', 
            type=str, 
            action='store', 
            required=True,
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
          model_name:str,
          node_type:str, 
          modality_transform_method: str,
          node_init_method:str,
          ckpt_path:str = None,
          ):
    seed_everything(train_settings.SEED)

    if node_init_method == "random":
        modality_transform_method = None

    # Process data
    process_node = ['gene/protein'] if node_type == "gene" else [node_type]

    node_encoder, embed_dim = NodeEncoderFactory.create_encoder(
            node_init_method=node_init_method,
            entity_type=node_type,
        )

    data_module = PrimeKGModule(
        process_node_lst=process_node,
        encoder=node_encoder
    )

    data_module.setup(stage="split", embed_dim=embed_dim)

    gcl_kwargs = {
        "in_dim": embed_dim,
        "hidden_dim": gcl_settings.GCL_HIDDEN_DIM,
        "out_dim": node_settings.GCL_TRAINED_NODE_DIM,
        "num_hidden_layers": gcl_settings.GCL_NUM_HIDDEN,
        "scheduler_type": train_settings.SCHEDULER_TYPE,
        "learning_rate": train_settings.LEARNING_RATE,
        "warm_up_ratio": train_settings.WARM_UP_RATIO,
        "modality_transform_method": modality_transform_method,
    }

    # Initialize GCL module
    if model_name == "dgi":
        model = DGIModule(**gcl_kwargs)
    elif model_name == "grace":
        model = GRACEModule(**gcl_kwargs)
    elif model_name == "ggd":
        model = GGDModule(**gcl_kwargs)
    else:
        raise NotImplementedError

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
        log_name = f"{model_name}_{node_type}_{modality_transform_method}_{str(int(time.time()))}"
        exp_name = log_name
        ckpt_dir = os.path.join(train_settings.OUT_DIR, "gcl", node_type, log_name)
        log_dir = os.path.join(train_settings.LOG_DIR, "gcl", node_type, log_name)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Setup callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            monitor="val_loss", 
            save_top_k=3, 
            mode="min",
            save_last=True,
            )
        
        early_stopping = EarlyStopping(monitor="val_loss", mode="min")

        logger = CometLogger(
            api_key=find_comet_api_key(),
            project_name=f"BioMedKG-GCL-{node_type}",
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
            train_dataloaders=data_module.train_dataloader(loader_type="neighbor"),
            val_dataloaders=data_module.val_dataloader(loader_type="neighbor"),
            ckpt_path=ckpt_path 
        )

    # Test
    elif task == "test":
        assert ckpt_path is not None, "Please specify checkpoint path."
        trainer = Trainer(**trainer_args)
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="neighbor"),
             ckpt_path=ckpt_path,
        )
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(**vars(parse_opt()))