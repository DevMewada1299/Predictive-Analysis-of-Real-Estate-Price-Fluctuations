import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # Input to embedding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, seq_length, embed_dim)  # Positional encoding
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)  # Output layer

    def forward(self, x):
        # Embed input features and add positional encoding
        x = self.embedding(x) + self.positional_encoding
        # Pass through transformer encoder
        x = self.encoder(x)
        # Output layer
        x = self.fc_out(x)
        return x
    
# Define parameters
input_dim = 5            # Number of input features in X
embed_dim = 64           # Embedding dimension for transformer
num_heads = 8            # Number of attention heads
ff_dim = 128             # Feedforward network dimension
num_layers = 4           # Number of transformer encoder layers
output_dim = 1           # Predicting one value per timestep
seq_length = 293         # Length of the sequence (timesteps)

# Instantiate the transformer model
transformer = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, seq_length)
