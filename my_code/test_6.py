# test_6 improvements on tes_5:
# 1. add a confusion matrix to better observe clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE
import seaborn as sns
import joblib


# Enable GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load denoised time series data
def load_denoised_data(file_paths):
    data = []
    for file in file_paths:
        df = pd.read_csv(file)
        data.append(df.values)
    return np.concatenate(data, axis=0)

# Define file paths
denoised_files = [
    "processed_data/denoised_steel.csv",
    "processed_data/denoised_roasted_steel.csv",
    "processed_data/denoised_aluminum.csv",
    "processed_data/denoised_brass.csv"
]

# Load and normalize data
time_series_data = load_denoised_data(denoised_files)
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)

# Apply PCA to reduce feature dimensions from 10000 to 396
pca = PCA(n_components=396)
time_series_data = pca.fit_transform(time_series_data)

# Reshape data for LSTM Autoencoder
sequence_length = 50  # Define a suitable sequence length
X = []
for i in range(len(time_series_data) - sequence_length):
    X.append(time_series_data[i: i + sequence_length])
X = np.array(X)

# Define LSTM Autoencoder
latent_dim = 16
input_layer = Input(shape=(sequence_length, X.shape[2]))
encoded = LSTM(latent_dim, return_sequences=False, activation="tanh")(input_layer)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(X.shape[2], return_sequences=True, activation="tanh")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = autoencoder.fit(X, X, epochs=100, batch_size=64, validation_split=0.1, verbose=1, callbacks=[early_stopping])

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
threshold = np.percentile(reconstruction_error, 95)  # Use 95th percentile

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
feature_data = pd.read_csv("processed_data/feature_data.csv")
true_labels = feature_data.iloc[:, 0].values  # Assuming first column is material labels
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

# Convert true labels to numerical format
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_features)

# Create Cluster-to-Material Mapping
cluster_material_mapping = {}
for cluster_id in np.unique(labels):
    mask = (labels == cluster_id)
    most_common_material = pd.Series(true_labels[mask]).mode()[0]  # Find most common material in cluster
    cluster_material_mapping[cluster_id] = most_common_material

# Print mapping results
print("Cluster to Material Mapping:")
for cluster, material in cluster_material_mapping.items():
    print(f"Cluster {cluster} → Material: {material}")

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels_encoded, labels)
class_labels = label_encoder.classes_  
cluster_labels = sorted(np.unique(labels))  # K-Means 

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=cluster_labels)
plt.xlabel("Predicted Cluster")
plt.ylabel("True Material Label")
plt.title("Confusion Matrix for K-Means Clustering")
plt.show()

# Use t-SNE for visualization
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_features)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.title("K-Means Clustering with t-SNE")
plt.show()
