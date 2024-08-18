import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from biomedkg.modules.utils import parameters_count

class AttentionFusion(nn.Module):
    def __init__(self,
                 embed_dim : int,
                 norm : bool = True,
                 ):
        super().__init__()

        self.norm = norm
        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, 
                x : torch.tensor, 
                ) -> torch.tensor:

        batch_size = x.size(0)

        x = x.view(batch_size, -1, self.embed_dim)
        
        if self.norm:
            x = F.normalize(x, dim=-1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
        )
        
        return x

class ReDAF(nn.Module):
    def __init__(self, embed_dim: int, num_modalities: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.modal_weights = nn.Parameter(torch.ones(num_modalities, 1, embed_dim))  # Updated shape [num_modalities, 1, embed_dim]
        self.sub_type_embeddings = nn.Embedding(num_modalities, embed_dim)

        self.transform_layer = nn.Linear(embed_dim, embed_dim)
        self.relational_context_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x, relational_context=0.2, sub_type_ids=None):
        device = x.device
        batch_size = x.size(0)

        x = x.view(batch_size, self.num_modalities, self.embed_dim)  # [batch_size, num_modalities, embed_dim]

        # Prepare relational context tensor
        relational_context = torch.full((1, self.embed_dim), relational_context, device=device)
        zeta_r = torch.sigmoid(self.relational_context_layer(relational_context))
        zeta_r = zeta_r.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, embed_dim]
        zeta_r = zeta_r.expand(batch_size, self.num_modalities, 1, self.embed_dim)  # [batch_size, num_modalities, 1, embed_dim]

        if sub_type_ids is not None:
            sub_type_embs = self.sub_type_embeddings(sub_type_ids).to(device)
        else:
            sub_type_embs = torch.zeros(batch_size, self.num_modalities, self.embed_dim, device=device)

        transformed_input = self.transform_layer(x + sub_type_embs)
        transformed_input = self.activation(transformed_input)

        # Broadcasting modal_weights across the batch size
        modal_weights_broadcasted = self.modal_weights.expand(self.num_modalities, batch_size, self.embed_dim).transpose(0, 1)
        # Now modal_weights_broadcasted has shape [batch_size, num_modalities, embed_dim]

        # Apply modal weights and relational context temperature, ensuring proper broadcasting
        weighted_input = transformed_input * modal_weights_broadcasted * zeta_r.squeeze(2)  # Squeeze out the singleton dimension for zeta_r
        
        h_joint = self.dropout(weighted_input)
        h_joint = self.activation(h_joint)

        return h_joint

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.tensor) -> torch.tensor:
        """
        x of shape [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

if __name__ == "__main__":
    embed_dim = 768
    num_modality = 2

    # model = AttentionFusion(
    #     embed_dim=embed_dim,
    #     norm=True,
    # )

    model = ReDAF(
        embed_dim= embed_dim,
    )

    print(model)

    print(f"Total parameters: {parameters_count(model):,}")

    x = torch.rand(1, embed_dim * num_modality)
    out = model(x)
    print(out.size())

    batch_size = 16
    x = torch.rand(batch_size, 1, embed_dim * num_modality)
    out = model(x)

    print(out.size())
