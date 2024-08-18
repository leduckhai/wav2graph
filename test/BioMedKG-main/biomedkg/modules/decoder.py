import math
import torch
import torch.nn.functional as F


class Decoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.rel_emb = torch.nn.Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)
    
    def forward(self, z, edge_index, edge_type):
        raise NotImplementedError
    

class TransE(Decoder):
    def __init__(self, num_relations, hidden_channels):
        super().__init__(num_relations, hidden_channels)
    
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.rel_emb, -bound, bound)
        with torch.no_grad():
            self.rel_emb.data = F.normalize(self.rel_emb.data, p=2, dim=-1)
            
    def forward(self, z, edge_index, edge_type):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]

        head = F.normalize(head, p=1.0, dim=-1)
        tail = F.normalize(tail, p=1.0, dim=-1)
        
        return -((head + rel) - tail).norm(p=1.0, dim=-1)


class DistMult(Decoder):
    def __init__(self, num_relations, hidden_channels):
        super().__init__(num_relations, hidden_channels)
    
    def forward(self, z, edge_index, edge_type):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]

        return torch.sum(head * rel * tail, dim=1)
    
class ComplEx(Decoder):
    def __init__(self, num_relations, hidden_channels):
        self.rel_emb_imag = torch.nn.Parameter(torch.empty(num_relations, hidden_channels))

        super().__init__(num_relations, hidden_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)
        torch.nn.init.xavier_uniform_(self.rel_emb_imag)

    def forward(self, z, edge_index, edge_type):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel_real = self.rel_emb[edge_type]
        rel_imag = self.rel_emb_imag[edge_type]

        head_real, head_imag = torch.chunk(head, 2, dim=-1)
        tail_real, tail_imag = torch.chunk(tail, 2, dim=-1)

        # Compute the real part of the Hermitian dot product
        score_real = (head_real * rel_real - head_imag * rel_imag) * tail_real
        score_real += (head_real * rel_imag + head_imag * rel_real) * tail_imag

        # Sum over the embedding dimension
        return torch.sum(score_real, dim=-1)


# Example usage:
if __name__ == "__main__":

    def test_model(model, num_nodes, num_relations, hidden_channels):
        # Create dummy data
        z = torch.randn((num_nodes, hidden_channels))
        edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        edge_type = torch.tensor([0, 1], dtype=torch.long)
        
        # Run the model
        output = model(z, edge_index, edge_type)
        print(f"Output of {model.__class__.__name__}: {output}")

    # Parameters
    num_nodes = 10
    num_relations = 5
    hidden_channels = 16

    # Initialize models
    transe = TransE(num_relations, hidden_channels)
    distmult = DistMult(num_relations, hidden_channels)
    complex_model = ComplEx(num_relations, hidden_channels)  

    # Test models
    print("Testing TransE model:")
    test_model(transe, num_nodes, num_relations, hidden_channels)

    print("\nTesting DistMult model:")
    test_model(distmult, num_nodes, num_relations, hidden_channels)

    print("\nTesting ComplEx model:")
    test_model(complex_model, num_nodes, num_relations, hidden_channels)