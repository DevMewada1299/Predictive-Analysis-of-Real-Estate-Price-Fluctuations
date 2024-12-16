import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn as sks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from Model.test import plottingData, getRMSE
from sklearn.metrics import mean_squared_error
import os


data =  plottingData()

y_tensor_test = data[0]
predicted_values_test = data[1] 
window_size = 5

# Sample chanel plotting for num_cities mentioned in test set.
base_folder = os.path.join(os.getcwd(), 'plots')
os.makedirs(base_folder, exist_ok=True)  

# Adjust the number of cities based on here to check perdictions and RMSE values -> saved plots at "plots/"
num_cities = 10
for i in range(num_cities):
    plt.figure(figsize=(10, 5))
    plt.plot(y_tensor_test.numpy()[:][i], label = 'Ground Truth')
    data = predicted_values_test[i][:].numpy()
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data.flatten(), kernel, mode='valid')
    rmse = mean_squared_error(y_tensor_test[0,:].numpy(),predicted_values_test[0,:].numpy())
    plt.plot(smoothed_data, label='Predicted Value')
    plt.title(f'Channel {i+1}')
    plt.xlabel('Time Steps, overall rmse : ' + str(rmse))
    plt.ylabel('Values')
    plt.legend()
    save_path = os.path.join(base_folder, f'my_plot_{i+1}.png')
    print(f"Attempting to save plot to: {save_path}") 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()