import os
from torch_geometric.nn import GAE
from biomedkg.configs import node_settings, data_settings
from biomedkg.modules import RGCN, RGAT, DistMult, TransE, ComplEx
from biomedkg.modules.fusion import AttentionFusion, ReDAF
from biomedkg.modules.node import EncodeNode, EncodeNodeWithModality

class ModalityFuserFactory:
    @staticmethod
    def create_fuser(method: str):
        if method == "attention":
            return AttentionFusion(
                    embed_dim=node_settings.PRETRAINED_NODE_DIM,
                    norm=True,
                )
        elif method == "redaf":
            return ReDAF(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                num_modalities = 2,
            )     
        else:
            return None

class KGEModelFactory:
    @staticmethod
    def create_model(
        encoder_name:str, 
        decoder_name:str,
        in_dim: int,
        hidden_dim:int, 
        out_dim: int,
        num_hidden_layers:int, 
        num_relation:int,
        num_heads:int = None,
        ):

        encoder = KGEModelFactory._get_encoder(
            encoder_name=encoder_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_hidden_layers=num_hidden_layers,
            num_relation=num_relation,
            num_heads=num_heads,
        )

        decoder = KGEModelFactory._get_decoder(
            decoder_name=decoder_name,
            num_relation=num_relation,
            hidden_channels=out_dim,
        )

        return GAE(
            encoder=encoder,
            decoder=decoder,
        )

    @staticmethod
    def _get_encoder(
        encoder_name:str,
        in_dim: int,
        hidden_dim:int, 
        out_dim: int,
        num_hidden_layers:int, 
        num_relation:int,
        num_heads:int = None,
        ):
        if encoder_name == "rgcn":
            return RGCN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_relations=num_relation
            )
        
        if encoder_name == "rgat":
            return RGAT(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_heads=num_heads,
                num_relations=num_relation
            )
    
    @staticmethod
    def _get_decoder(
        decoder_name: str,
        num_relation:int,
        hidden_channels:int,
    ):
        if decoder_name == "transe":
            return TransE(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )
        if decoder_name == "dismult":
            return DistMult(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )
        if decoder_name == "complex":
            return ComplEx(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )


class NodeEncoderFactory:
    @staticmethod
    def create_encoder(
            node_init_method:str,
            gcl_embed_path:str = None,
            entity_type:str = None,
            ):
        if node_init_method == "gcl":
            assert gcl_embed_path is not None
            encoder = EncodeNode(
                embed_path=gcl_embed_path
                )
            embed_dim = node_settings.GCL_TRAINED_NODE_DIM
        elif node_init_method == "llm":
            entity_names = list(data_settings.NODES_LST) if entity_type is None else entity_type
            encoder = EncodeNodeWithModality(
                entity_type=entity_names, 
                embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
                )
            embed_dim = node_settings.PRETRAINED_NODE_DIM
        elif node_init_method == "random":
            encoder = None
            embed_dim = node_settings.GCL_TRAINED_NODE_DIM
        
        return encoder, embed_dim