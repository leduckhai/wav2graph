import os
import glob
import torch
import pickle
from typing import List, Union
from pathlib import Path
from biomedkg.configs import node_settings

class EncodeNode:
    def __init__(
        self,
        embed_path : str,
        ):
        assert os.path.exists(embed_path), f"Can't find {embed_path}."

        self.embed_path = embed_path

        self.node_embed = self._get_feature_embedding_dict()
    
    def __call__(self, node_name_lst:List[str]) -> torch.tensor:
        node_embedding = []

        for node_name in node_name_lst:

            # Get embeddings by name
            modality_embedding = self.node_embed.get(node_name, None)

            if modality_embedding is not None:
                node_embedding.append(torch.tensor(modality_embedding))
            else:
                node_embedding.append(torch.rand(node_settings.GCL_TRAINED_NODE_DIM))
                    
        node_embedding = torch.stack(node_embedding, dim=0)
        return node_embedding

    def _get_feature_embedding_dict(self,):
        node_embed = dict()

        pickle_files = glob.glob(self.embed_path + "/*.pickle")
        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as file:
                data = pickle.load(file=file)
                node_embed.update(data)
        return node_embed


class EncodeNodeWithModality:
    def __init__(
        self,
        entity_type : Union[str, List[str]],
        embed_path : str,
        ):
        assert os.path.exists(embed_path), f"Can't find {embed_path}."

        if isinstance(entity_type, str):
            assert entity_type in ["gene", "disease", "drug"]
            entity_type = [entity_type]
        elif isinstance(entity_type, List):
            for idx in range(len(entity_type)):
                if entity_type[idx].startswith("gene"):
                    entity_type[idx] = "gene"
            
            assert sorted(entity_type) ==  sorted(["gene", "disease", "drug"])


        self.entity_type = entity_type
        self.embed_path = embed_path

        self.node_embedding_map = self._get_feature_embedding_dict()
    
    def __call__(self, node_name_lst:List[str]) -> torch.tensor:
        node_embedding = []

        for node_name in node_name_lst:
            embedding = self.node_embedding_map.get(node_name, None)
            if embedding is None:
                embedding = torch.rand(node_settings.PRETRAINED_NODE_DIM * 2)
            node_embedding.append(embedding)

        node_embedding = torch.stack(node_embedding, dim=0)
        return node_embedding

    def _get_feature_embedding_dict(self,) -> dict:
        node_embedding_map = dict()

        for entity_type in self.entity_type:
            
            modality_dict = dict()
            all_modality = set()

            # Load LLM embeddings by modality and store them in dictionary
            pickle_files = glob.glob(self.embed_path + f"/{entity_type}*.pickle")
            for pickle_file in pickle_files:
                modality_name = Path(pickle_file).stem.split("_")[1]
                with open(pickle_file, "rb") as file:
                    data = pickle.load(file=file)
                    if modality_name in modality_dict:
                        modality_dict[modality_name].update(data)
                    else:
                        modality_dict[modality_name] = data
                all_modality.add(modality_name)
            
            # Ensure 'seq' is always at the second position
            all_modality = sorted(all_modality, key=lambda s: (1, s) if "seq" in s else (0, s))
            
            # Get all node_name
            all_node_name = set()
            for modality_name in all_modality:
                all_node_name.update(modality_dict[modality_name].keys())
            
            # Get embedding by node name
            for node_name in all_node_name:
                fuse_embed = list()
                for modality_name in all_modality:
                    embedding = modality_dict[modality_name].get(node_name, None)
                    if embedding is not None:
                        fuse_embed.append(torch.tensor(embedding))
                    else:
                        fuse_embed.append(torch.rand(node_settings.PRETRAINED_NODE_DIM))

                node_embedding_map[node_name] = torch.cat(fuse_embed, dim=0)

        return node_embedding_map


if __name__ == "__main__":    
    encoder = EncodeNodeWithModality(
        entity_type="gene",
        embed_path="../../data/embed"
    )

    node_name_lst = ['PHYHIP', 'GPANK1', 'ZRSR2','NRF1','PI4KA','SLC15A1','EIF3I','FAXDC2','MT1A','SORT1']

    embeddings = encoder(node_name_lst)

    print(embeddings.size())

    encoder = EncodeNode(
        embed_path="../../data/gcl_embed"
    )

    node_name_lst = ["(1,2,6,7-3H)Testosterone",
                    "(4-{(2S)-2-[(tert-butoxycarbonyl)amino]-3-methoxy-3-oxopropyl}phenyl)methaneseleninic acid",
                    "(6R)-Folinic acid",
                    "(6S)-5,6,7,8-tetrahydrofolic acid",
                    "(R)-warfarin",
                    "(S)-2-Amino-3-(4h-Selenolo[3,2-B]-Pyrrol-6-Yl)-Propionic Acid",
                    "(S)-2-Amino-3-(6h-Selenolo[2,3-B]-Pyrrol-4-Yl)-Propionic Acid",
                    "(S)-Warfarin",
                    "1,10-Phenanthroline",
                    "1-Testosterone",
                    ]

    embeddings = encoder(node_name_lst)

    print(embeddings.size())
