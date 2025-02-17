# test_5 Improvements on test_4
# 1. Re-added Early stopping to allow training to stop early at the optimal point to prevent overfitting and avoid training loss going down but validation loss going up.
# 2. Adjust the anomaly threshold from mean + 3 * std to 95% quantile (np.percentile) to set a more stable threshold.
# 3. t-SNE used for dimensionality reduction visualization

# Version 3.0 can be used for training
# Analyze training results:
# 1. Training loss decreases steadily and finally approaches 0.79, converging well.
# Validation loss stabilizes at around 50 rounds, without any significant increase, indicating no obvious overfitting.
# EarlyStopping (priority=10) effectively avoids overfitting, and batch_size=32 also performs stably.
# 2. Errors are mainly centered between 0.39 - 0.44, but there are still some peaks.
# Anomaly threshold ~0.44 (95% quantile) is more robust and more accurate than mean + 3*std method.
# Only a small number of data points exceed the anomaly threshold (red dashed line), indicating that the anomaly detection is not overly sensitive.
# 3. K-Means clustering (t-SNE dimensionality reduction)
# The data distribution is clearer than the previous PCA 2D effect, and the distribution of different clusters is more separated.
# You can see the obvious three major types of clusters


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
from sklearn.manifold import TSNE


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

##########################################
# Improvement 1
##########################################

# Train Autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(X, X, epochs=70, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stopping])

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title("Training vs. Validation Loss")
plt.legend()
plt.show()

##################################
# Improvement 2
##################################

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
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

################################################
# Improvement 3
################################################

# Evaluate best K value using silhouette score
best_k = 4
best_score = -1
for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_features)
    score = silhouette_score(X_features, labels)
    print(f"K={k}, Silhouette Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

# Perform K-Means clustering with best k
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_features)

# Compute silhouette score
silhouette = silhouette_score(X_features, labels)
db_score = davies_bouldin_score(X_features, labels)
ch_score = calinski_harabasz_score(X_features, labels)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

# Use t-SNE for visualization
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_features)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.title("K-Means Clustering with t-SNE")
plt.show()
