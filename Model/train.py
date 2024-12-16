from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn as sks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import DataLoader,TensorDataset
import data
from Model.TransformerModel import TransformerModel
import data.preprocess_get_data

input_dim = 5            # Number of input features in X
embed_dim = 64           # Embedding dimension for transformer
num_heads = 8            # Number of attention heads
ff_dim = 128             # Feedforward network dimension
num_layers = 4           # Number of transformer encoder layers
output_dim = 1           # Predicting one value per timestep
seq_length = 293         # Length of the sequence (timesteps)

transformer = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, seq_length)

# Define loss and optimizer
criterion = nn.MSELoss()  # For regression tasks
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

dataset,dataloader = data.preprocess_get_data.get_tensor_dataset()

num_epochs = 15
for epoch in range(num_epochs):
    transformer.train()
    total_loss = 0

    # Initialize tqdm progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_X, batch_y in tepoch:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = transformer(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update tqdm bar with the current loss
            tepoch.set_postfix(loss=loss.item())

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

torch.save(transformer.state_dict(), 'saved_model/model.pth')