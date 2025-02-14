# Solution: test_3
# 1. Reduce dimension to 396 by PCA, time_series_data has shape (396, 10000)
# Change PCA dimension (n_components) mainly affects the compression level of data, information retention rate, and the computational performance of the model
# The number of samples is 396 and the number of features is 10000, so the upper limit is 396.
# 2. change the activation function of the LSTM layer from relu to tanh, so that Tensorflow will automatically use CuDNN for acceleration
# 3. Add an Earlystoping to avoid wasting resources, so that if the model doesn't improve within a few epochs, it will stop training automatically.
# monitor='val_loss': monitor validation loss
# patience=10: stop training if val_loss doesn't drop within 10 epochs.
# restore_best_weights=True: prevent overfitting, return to optimal weights

# This version is the one that is initially available for training.
# The training results are analyzed:
# 1. the training loss continues to decrease, the validation loss decreases and then stabilizes, indicating that the model is not overfitting.
# 2. but the validation loss is lower than the training loss, this usually indicates:
# The data is noisy, perhaps the model performs better on some inputs than the training data.
# Maybe EarlyStopping was triggered a little early and the model has not fully converged.
# 3. batch size is too small


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

##########################################
# Improvement 1
##########################################

# Apply PCA to reduce feature dimensions from 10000 to 500
pca = PCA(n_components=396)
time_series_data = pca.fit_transform(time_series_data)

# Reshape data for LSTM Autoencoder
sequence_length = 50  # Define a suitable sequence length
X = []
for i in range(len(time_series_data) - sequence_length):
    X.append(time_series_data[i: i + sequence_length])
X = np.array(X)

###########################################
# Improvement 2
###########################################

# Define LSTM Autoencoder
latent_dim = 16
input_layer = Input(shape=(sequence_length, X.shape[2]))
encoded = LSTM(latent_dim, return_sequences=False, activation="tanh")(input_layer)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(X.shape[2], return_sequences=True, activation="tanh")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

##########################################
# Improvement 3
##########################################

# Train Autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(X, X, epochs=50, batch_size=8, validation_split=0.1, verbose=1, callbacks=[early_stopping])

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
feature_data = pd.read_csv("processed_data/feature_data.csv")
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_features)

# Compute silhouette score
silhouette = silhouette_score(X_features, labels)
db_score = davies_bouldin_score(X_features, labels)
ch_score = calinski_harabasz_score(X_features, labels)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f} ")
print(f"Calinski-Harabasz Index: {ch_score:.4f} ")


# Plot clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_features[:, 0], X_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering of Materials")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Cluster Label')
plt.show()
