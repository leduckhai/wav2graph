import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch_geometric.utils import negative_sampling
from torchmetrics.wrappers import BootStrapper
from torchmetrics import MetricCollection, AUROC, AveragePrecision, F1Score
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from typing import Tuple
from biomedkg.factory import KGEModelFactory, ModalityFuserFactory
from biomedkg.configs import kge_settings, node_settings, train_settings
from biomedkg.modules.metrics import EdgeWisePrecision

class KGEModule(LightningModule):
    def __init__(self,
                encoder_name:str = kge_settings.KGE_ENCODER,
                decoder_name: str = kge_settings.KGE_DECODER,
                in_dim: int = 128,
                hidden_dim: int = kge_settings.KGE_HIDDEN_DIM,
                out_dim: int = node_settings.KGE_TRAINED_NODE_DIM,
                num_hidden_layers: int = kge_settings.KGE_NUM_HIDDEN,
                num_relation: int = 8,
                num_heads: int =kge_settings.KGE_NUM_HEAD,
                scheduler_type : str = train_settings.SCHEDULER_TYPE,
                learning_rate: float = train_settings.LEARNING_RATE,
                warm_up_ratio : float = train_settings.WARM_UP_RATIO,
                select_edge_type_id : int = None,
                neg_ratio: int = None,
                node_init_method:str = kge_settings.KGE_NODE_INIT_METHOD,
                modality_transform_method: str = node_settings.MODALITY_TRANSFORM_METHOD
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_embedding_dim = in_dim

        self.llm_init_node = False
        
        if node_init_method == "llm":
            self.llm_init_node = True
            self.modality_transform = ModalityFuserFactory.create_fuser(method=modality_transform_method)
        
        self.save_hyperparameters(
                {
                    "node_init_method": node_init_method,
                }
            )

        self.model = KGEModelFactory.create_model(
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_hidden_layers=num_hidden_layers,
            num_relation=num_relation,
            num_heads=num_heads
        )
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio

        metrics = MetricCollection(
            {
                "AUROC": BootStrapper(AUROC(task="binary"),),
                "AveragePrecision": BootStrapper(AveragePrecision(task="binary")),
                "F1": BootStrapper(F1Score(task="binary")),
            }
        )
        self._edge_index_map = dict()
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.select_edge_type_id = select_edge_type_id
        self.neg_ratio = neg_ratio
    
    def fuse_modality(self, x) -> torch.tensor:
        if self.llm_init_node:
            # Reshape if does not apply transformation
            if self.modality_transform is None:
                x = x.view(x.size(0), -1, self.feature_embedding_dim)
                x = F.normalize(x, dim=-1)
            else:
                x = self.modality_transform(x)

            x = torch.mean(x, dim=1)

        return x

    def sample_neg_edges(
            self, 
            edge_index: torch.tensor, 
            edge_type: torch.tensor
            ) -> Tuple[torch.tensor, torch.tensor]:
        if self.neg_ratio is None:
            neg_edge_index = negative_sampling(edge_index)
            neg_edge_type = edge_type
        else:
            neg_edge_index = negative_sampling(edge_index, num_neg_samples=self.neg_ratio*edge_index.size(-1))
            neg_edge_type = edge_type.repeat(self.neg_ratio)
            neg_edge_indices = torch.randperm(neg_edge_type.size(0))
            neg_edge_type = neg_edge_type[neg_edge_indices]
        return neg_edge_index, neg_edge_type
    
    def forward(self, x, edge_index, edge_type):
        x = self.fuse_modality(x=x)

        return self.encoder(x, edge_index, edge_type)
    
    def training_step(self, batch):
        x = self.fuse_modality(x=batch.x)

        if self.select_edge_type_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self.select_edge_type_id)

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(batch.edge_index, batch.edge_type)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.fuse_modality(x=batch.x)

        if self.select_edge_type_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self.select_edge_type_id)
      
        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(batch.edge_index, batch.edge_type)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.valid_metrics.update(pred, gt.to(torch.int32))
        if hasattr(self, "edge_wise_pre_valid"):
            self.edge_wise_pre_valid.update(pos_pred, batch.ededge_typeedge_typege)

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        
        if hasattr(self, "edge_wise_pre_valid"):
            edge_wise_pre = self.edge_wise_pre_valid.compute()
            self.log_dict(edge_wise_pre)
            self.edge_wise_pre_valid.reset()

        self.log_dict(output)
        self.valid_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        x = self.fuse_modality(x=batch.x)

        if self.select_edge_type_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self.select_edge_type_id)
            
        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(batch.edge_index, batch.edge_type)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.test_metrics.update(pred, gt.to(torch.int32))
        if hasattr(self, "edge_wise_pre_test"):
            self.edge_wise_pre_test.update(pos_pred, batch.edge_type)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()

        if hasattr(self, "edge_wise_pre_test"):
            edge_wise_pre = self.edge_wise_pre_test.compute()
            self.log_dict(edge_wise_pre)
            self.edge_wise_pre_test.reset()

        self.log_dict(output)
        self.test_metrics.reset()
        return output
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        scheduler = self._get_scheduler(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _get_scheduler(self, optimizer):
        scheduler_args = {
            "optimizer": optimizer,
            "num_training_steps": int(self.trainer.estimated_stepping_batches),
            "num_warmup_steps": int(self.trainer.estimated_stepping_batches * self.warm_up_ratio),
        }
        if self.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(**scheduler_args)
        if self.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(**scheduler_args)
    
    @property
    def edge_mapping(self): 
        return self._edge_index_map
       
    @edge_mapping.setter 
    def edge_mapping(self, edge_mapping_dict:dict): 
        self._edge_index_map = edge_mapping_dict
        self.edge_wise_pre_valid = EdgeWisePrecision(class_mapping=self._edge_index_map)
        self.edge_wise_pre_test = EdgeWisePrecision(class_mapping=self._edge_index_map)
