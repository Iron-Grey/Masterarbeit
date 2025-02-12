import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tensorflow.keras.callbacks import EarlyStopping



# Load denoised time series data
def load_denoised_data(file_paths):
    data = []
    for file in file_paths:
        df = pd.read_csv(file)
        data.append(df.values)
    return np.concatenate(data, axis=0)

# Define file paths
denoised_files = [
    "C:/Users/c1257/Desktop/processed_data/denoised_steel.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_roasted_steel.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_aluminum.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_brass.csv"
]

# Load and normalize data
time_series_data = load_denoised_data(denoised_files)
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)

print("Shape of time_series_data:", time_series_data.shape)
pca = PCA(n_components=396)
time_series_data = pca.fit_transform(time_series_data)
print(f"Explained variance ratio sum: {sum(pca.explained_variance_ratio_):.4f}")