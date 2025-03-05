import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import yaml
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout, LeakyReLU
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.cluster import KMeans
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import mode
from sklearn.manifold import TSNE

# Enable GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -------------------- Read Configuration File --------------------
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Load config parameters
sequence_length = config["data"]["sequence_length"]
file_paths = config["data"]["file_paths"]
feature_data_path = config["data"]["feature_data_path"]
pca_components = config["data"]["pca_components"]
latent_dim = config["model"]["latent_dim"]
dropout_rate = config["model"]["dropout_rate"]
lstm_units_encoder_first = config["model"]["lstm_units"]["encoder"]["first"]
lstm_units_encoder_second = config["model"]["lstm_units"]["encoder"]["second"]
lstm_units_decoder_first = config["model"]["lstm_units"]["decoder"]["first"]
lstm_units_decoder_second = config["model"]["lstm_units"]["decoder"]["second"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
validation_split = config["training"]["validation_split"]
patience = config["training"]["patience"]

# -------------------- Data Loading and Preprocessing --------------------
def load_denoised_data(file_paths):
    data = []
    for file in file_paths:
        df = pd.read_csv(file)
        print(f"Loaded {file}, shape: {df.shape}")  # Check data shape
        data.append(df.values)
    return np.concatenate(data, axis=0)


# Load and normalize data
time_series_data = load_denoised_data(file_paths)
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)

# Apply PCA to reduce dimensions from 10000 to 396
pca = PCA(n_components=pca_components)
time_series_data = pca.fit_transform(time_series_data)

X = []
for i in range(len(time_series_data) - sequence_length):
    X.append(time_series_data[i: i + sequence_length])
X = np.array(X)

# -------------------- Build LSTM Autoencoder --------------------
input_layer = Input(shape=(sequence_length, X.shape[2]))
# Encoder
encoded = LSTM(lstm_units_encoder_first, return_sequences=True, activation="tanh")(input_layer)
encoded = Dropout(dropout_rate)(encoded)
encoded = LSTM(lstm_units_encoder_second, return_sequences=False, activation="tanh")(encoded)
encoded = Dropout(dropout_rate)(encoded)
encoded = Dense(latent_dim, name='encoder_dense')(encoded)
encoded = LeakyReLU(alpha=0.1, name='encoder_output')(encoded)

# Decoder
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(lstm_units_decoder_first, return_sequences=True, activation="tanh")(decoded)
decoded = Dropout(dropout_rate)(decoded)
decoded = LSTM(X.shape[2], return_sequences=True, activation='tanh', name='decoder_output')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=patience,
                               restore_best_weights=True)
history = autoencoder.fit(X, X,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_split=validation_split,
                          verbose=1,
                          callbacks=[early_stopping]
                          )

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title("Training vs. Validation Loss")
plt.legend()
plt.show()

# -------------------- Extract Autoencoder Outputs --------------------
# Get decoded output (reconstructed data)
X_pred = autoencoder.predict(X)
# Convert 3D time series data to 2D by taking the mean over time steps
decoded_features = X_pred.mean(axis=1)

# Build encoder model to extract latent features
encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(X)

# -------------------- Load Original Clustering Feature Data --------------------
feature_data = pd.read_csv(feature_data_path)
true_labels = feature_data.iloc[:, 0].values  # Assume first column is material labels
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

# Convert labels to numerical encoding
label_encoder = LabelEncoder()
true_labels_encoded_full = label_encoder.fit_transform(true_labels)

# Use the first len(X) samples to correspond with autoencoder data (assuming the same order)
true_labels_auto = true_labels[:len(X)]
true_labels_encoded_auto = label_encoder.transform(true_labels_auto)

# Set clustering parameters
n_clusters = 4
random_state = 42

# -------------------- Clustering 1: Original Features Clustering --------------------
kmeans_original = KMeans(n_clusters=n_clusters, random_state=random_state)
cluster_labels_original = kmeans_original.fit_predict(X_features)

# Map clusters to true labels using mode
new_labels_original = np.zeros_like(cluster_labels_original)
cluster_to_material_original = {}
for cluster_id in range(n_clusters):
    mask = (cluster_labels_original == cluster_id)
    if np.any(mask):
        most_common_material = mode(true_labels_encoded_full[mask])[0][0]
        cluster_to_material_original[cluster_id] = most_common_material
        new_labels_original[mask] = most_common_material

sil_original = silhouette_score(X_features, cluster_labels_original)

# -------------------- Clustering 2: Decoded Data Clustering --------------------
kmeans_decoded = KMeans(n_clusters=n_clusters, random_state=random_state)
cluster_labels_decoded = kmeans_decoded.fit_predict(decoded_features)

new_labels_decoded = np.zeros_like(cluster_labels_decoded)
cluster_to_material_decoded = {}
for cluster_id in range(n_clusters):
    mask = (cluster_labels_decoded == cluster_id)
    if np.any(mask):
        most_common_material = mode(true_labels_encoded_auto[mask])[0][0]
        cluster_to_material_decoded[cluster_id] = most_common_material
        new_labels_decoded[mask] = most_common_material

sil_decoded = silhouette_score(decoded_features, cluster_labels_decoded)

# -------------------- Clustering 3: Encoded Data Clustering --------------------
kmeans_encoded = KMeans(n_clusters=n_clusters, random_state=random_state)
cluster_labels_encoded = kmeans_encoded.fit_predict(encoded_features)

new_labels_encoded = np.zeros_like(cluster_labels_encoded)
cluster_to_material_encoded = {}
for cluster_id in range(n_clusters):
    mask = (cluster_labels_encoded == cluster_id)
    if np.any(mask):
        most_common_material = mode(true_labels_encoded_auto[mask])[0][0]
        cluster_to_material_encoded[cluster_id] = most_common_material
        new_labels_encoded[mask] = most_common_material

sil_encoded = silhouette_score(encoded_features, cluster_labels_encoded)

# -------------------- Clustering Comparison Results --------------------
print("Silhouette Score:")
print(f"Original Features: {sil_original:.4f}")
print(f"Decoded Data: {sil_decoded:.4f}")
print(f"Encoded Data: {sil_encoded:.4f}")

# Compute confusion matrices
conf_matrix_original = confusion_matrix(true_labels_encoded_full, new_labels_original)
conf_matrix_decoded = confusion_matrix(true_labels_encoded_auto, new_labels_decoded)
conf_matrix_encoded = confusion_matrix(true_labels_encoded_auto, new_labels_encoded)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(conf_matrix_original, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix: Original Features")
plt.xlabel("Predicted Material")
plt.ylabel("True Material")

plt.subplot(1, 3, 2)
sns.heatmap(conf_matrix_decoded, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix: Decoded Data")
plt.xlabel("Predicted Material")
plt.ylabel("True Material")

plt.subplot(1, 3, 3)
sns.heatmap(conf_matrix_encoded, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix: Encoded Data")
plt.xlabel("Predicted Material")
plt.ylabel("True Material")

plt.tight_layout()
plt.show()

# -------------------- t-SNE Visualization of Clustering Results --------------------
perplexity = 30

# Perform t-SNE dimensionality reduction to 2D for each feature set
tsne_original = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(X_features)
tsne_decoded = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(decoded_features)
tsne_encoded = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(encoded_features)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(tsne_original[:, 0], tsne_original[:, 1], c=cluster_labels_original, cmap='viridis', alpha=0.7,
            edgecolors='k')
plt.title("t-SNE: Original Features Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.subplot(1, 3, 2)
plt.scatter(tsne_decoded[:, 0], tsne_decoded[:, 1], c=cluster_labels_decoded, cmap='viridis', alpha=0.7, edgecolors='k')
plt.title("t-SNE: Decoded Data Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.subplot(1, 3, 3)
plt.scatter(tsne_encoded[:, 0], tsne_encoded[:, 1], c=cluster_labels_encoded, cmap='viridis', alpha=0.7, edgecolors='k')
plt.title("t-SNE: Encoded Data Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.show()
