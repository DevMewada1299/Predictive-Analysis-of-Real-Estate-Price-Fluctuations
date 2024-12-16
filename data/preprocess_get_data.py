import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn as sks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import DataLoader,TensorDataset

df = pd.read_csv('data/raw data/ZHVInew.txt')

df = df.drop(columns=['StateName','RegionID','Metro','RegionType'], inplace= False)

df = df.drop(columns=['SizeRank','RegionName','State','CountyName'])

df = df.interpolate(method='linear', axis=0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

train_data_perc = int(0.8*21531)

train_data = scaled_data[:train_data_perc,:]
test_data  = scaled_data[train_data_perc:,:]

# Taking limited train data for demonstration
train_data = scaled_data[:train_data_perc,:]
test_data  = scaled_data[train_data_perc:,:]

train_data_lim = train_data[:1000,:]
test_data_lim = test_data[:1000,:]

window_size = 5
stride = 1

# Create sliding windows
def create_sliding_windows(data, window_size, stride):
    C, T = data.shape
    X = []
    y = []
    for start in range(0, T - window_size):
        end = start + window_size
        X.append(data[:, start:end])
        y.append(data[:, end])  # Target is the next time step after the window
    return np.array(X), np.array(y)

X, y = create_sliding_windows(train_data_lim, window_size, stride)


# Reshaping y to match the size 
y = y.reshape(293, 1000, 1)

def get_prepared_data():
    # Swap axes to match (batch_size, sequence_length, features)
    X_prepared = np.transpose(X, (1, 0, 2))  # Shape: (128, 293, 5)
    y_prepared = np.transpose(y, (1, 0, 2))  # Shape: (128, 293, 1)

    return X_prepared, y_prepared

def convert_to_tensor():

    X_prepared, y_prepared = get_prepared_data()
    X_tensor = torch.tensor(X_prepared, dtype=torch.float32)
    y_tensor = torch.tensor(y_prepared, dtype=torch.float32)

    return X_tensor, y_tensor


def get_tensor_dataset():

    X_tensor, y_tensor = convert_to_tensor()
    dataset = TensorDataset(X_tensor,y_tensor)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=False)

    return dataset,dataloader

# For Test Data

def test_data():
    X_test, y_test = create_sliding_windows(test_data_lim, window_size, stride)
    y_test = y_test.reshape(293, 1000, 1)

    X_prepared_test = np.transpose(X_test, (1, 0, 2))  # Shape: (128, 293, 5)
    y_prepared_test = np.transpose(y_test, (1, 0, 2))  # Shape: (128, 293, 1)

    X_tensor_test = torch.tensor(X_prepared_test, dtype=torch.float32) 
    y_tensor_test = torch.tensor(y_prepared_test, dtype=torch.float32)

    return X_tensor_test, y_tensor_test



