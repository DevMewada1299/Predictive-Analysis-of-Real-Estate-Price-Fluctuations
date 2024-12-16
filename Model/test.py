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
from sklearn.metrics import root_mean_squared_error, mean_squared_error

input_dim = 5            # Number of input features in X
embed_dim = 64           # Embedding dimension for transformer
num_heads = 8            # Number of attention heads
ff_dim = 128             # Feedforward network dimension
num_layers = 4           # Number of transformer encoder layers
output_dim = 1           # Predicting one value per timestep
seq_length = 293         # Length of the sequence (timesteps)

transformer = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, seq_length)
transformer.load_state_dict(torch.load('saved_model/model.pth'))

X_tensor_test, y_tensor_test = data.preprocess_get_data.test_data()

transformer.eval()
with torch.no_grad():
    predicted_values_test = transformer(X_tensor_test)  # Shape: (batch_size, sequence_length, output_dim)

    print("Predicted values shape:", predicted_values_test.shape)

def getRMSE():

    rmse = mean_squared_error(y_tensor_test[0,:].numpy(),predicted_values_test[0,:].numpy())

    return rmse


# Storing data for Plotting
def plottingData():

    stored_data_for_plotting = {}

    stored_data_for_plotting[0] = y_tensor_test
    stored_data_for_plotting[1] = predicted_values_test

    return stored_data_for_plotting








