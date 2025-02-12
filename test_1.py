# 这边的问题是GPU内存不足，无法分配tensor形状[10000, 40000]的张量
# 改进：test_2
#    1. 降低Batch size
#    2. 用memory_growth限制GPU内存增长,默认情况下，TensorFlow 会在 GPU 上 一次性占满所有可用显存，即使模型不需要那么多内存。


# The problem on this side is that there is not enough GPU memory to allocate the tensor shape [10000, 40000] tensor


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


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

# Reshape data for LSTM Autoencoder
sequence_length = 50  # Define a suitable sequence length
X = []
for i in range(len(time_series_data) - sequence_length):
    X.append(time_series_data[i: i + sequence_length])
X = np.array(X)

# Define LSTM Autoencoder
latent_dim = 16
input_layer = Input(shape=(sequence_length, X.shape[2]))
encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(input_layer)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(X.shape[2], activation='relu', return_sequences=True)(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
history = autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# 绘制训练和验证损失曲线，评估自编码器训练效果
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title("Training vs. Validation Loss")
plt.legend()
plt.show()

# Compute reconstruction errors
X_pred = autoencoder.predict(X)
reconstruction_error = np.mean(np.abs(X - X_pred), axis=(1, 2))
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# Detect anomalies
anomalies = reconstruction_error > threshold

# Plot reconstruction error
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label='Reconstruction Error')
plt.axvline(threshold, color='r', linestyle='dashed', label='Anomaly Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

# Load feature data for clustering
feature_data = pd.read_csv("C:/Users/c1257/Desktop/processed_data/feature_data.csv")
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_features)

# Compute silhouette score
silhouette = silhouette_score(X_features, labels)
db_score = davies_bouldin_score(X_features, labels)
ch_score = calinski_harabasz_score(X_features, labels)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}  (越小越好)")
print(f"Calinski-Harabasz Index: {ch_score:.4f}  (越大越好)")


# Plot clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_features[:, 0], X_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering of Materials")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Cluster Label')
plt.show()