import os
import yaml
import glob
import pickle
import argparse
import pandas as pd
from typing import List
from pathlib import Path
from tqdm.auto import tqdm

from biomedkg.modules.utils import generator
from biomedkg.modules.embed import NodeEmbedding

def get_feature(
        file_name : str,
        idetifier_column: str,
        modality_columns : List[str],
        model_name_for_each_modality : List[str],
        save_files : List[str],
        batch_size: int = 16,
):

    df = pd.read_csv(file_name)

    for modality, model_name, save_file in zip(modality_columns, model_name_for_each_modality, save_files):

        print(f"Process {modality} with {model_name}")
        
        sub_df = df[[idetifier_column, modality]]
        sub_df = sub_df.dropna()
        sub_df = sub_df.drop_duplicates()

        modality_ids = sub_df[idetifier_column].to_list()

        entity_name_lst = list()
        entity_feature_lst = list()

        node_embeder = NodeEmbedding(model_name_or_path=model_name)

        with tqdm(total=len(modality_ids), desc=f"Processing {modality}") as pbar:
            for identity in generator(modality_ids, batch_size):
                modality_feature = sub_df[sub_df[idetifier_column].isin(identity)]
                
                entity_names = modality_feature[idetifier_column].to_list()
                entity_features = modality_feature[modality].to_list()

                hidden_state = node_embeder(entity_features)
                
                entity_name_lst.extend(entity_names)
                entity_feature_lst.extend(hidden_state.numpy())
                
                pbar.update(len(identity))  
                        
        feature_dict = dict(zip(entity_name_lst, entity_feature_lst))
        
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        with open(save_file, "wb") as file:
            pickle.dump(feature_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        del node_embeder

        import gc
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', "-f", type=str, help='The path to the YAML file')
    args = parser.parse_args()

    yaml_file = args.yaml_file

    with open(yaml_file, "r") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    if "primekg" in yaml_file:
        
        for entity_type in data.keys():
            if entity_type == "gene/protein":
                gene_protein = data[entity_type]
                for gene_type in gene_protein.keys():
                    entity = gene_protein[gene_type]
                    get_feature(**entity)
            else:
                entity = data[entity_type]
                get_feature(**entity)

        # Merge amino_acid and dna for gene/protein  
        embed_path = "data/embed"
        pickle_files = glob.glob(embed_path + "/protein*pickle")  

        protein_desc = dict()
        protein_seq = dict()

        for pickle_file in pickle_files:
            file_name = Path(pickle_file).stem
            modality_name = file_name.split("_")[-2]

            with open(pickle_file, "rb") as file:
                data = pickle.load(file=file)

            if modality_name == "desc":
                protein_desc.update(data)
            else:
                protein_seq.update(data)

        with open(os.path.join(embed_path, "gene_seq_proteinbert_dna_bert.pickle"), "wb") as file:
            pickle.dump(protein_seq, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(embed_path, "gene_desc_biobert.pickle"), "wb") as file:
            pickle.dump(protein_desc, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:    
        for entity_type in data.keys():
            entity = data[entity_type]
            get_feature(**entity)