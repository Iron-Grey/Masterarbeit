# test_7 improvements on test_6
# 1. patience=10 may be too short for training early stops, try patience=15
# 2. add Dense layer to encoded layer to make encoding more compact
# 3. try to calculate anomaly thresholds separately by material class
# 4. the plot of clustering only shows 3 categories, we have 4 materials in total, suggesting that there is a possibility that two materials are highly overlapping
# Try to change tsne dimensionality reduction to down to 3 dimensions or use DBSCAN as an alternative
# 5. since K-Means has no supervisory information, it does not automatically tell you which cluster corresponds to which material, you need to determine the actual material category (steel, aluminum, brass, etc.) that corresponds to each cluster category (cluster label)
# 6. add code to save the model after it passes the final round of testing, so that the model can be put to the test directly.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score, confusion_matrix
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

# Load denoised time series data
def load_denoised_data(file_paths):
    data = []
    for file in file_paths:
        df = pd.read_csv(file)
        print(f"Loaded {file}, shape: {df.shape}")  # 检查数据形状
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
encoded = Dense(latent_dim // 2, activation="tanh")(encoded)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(X.shape[2], return_sequences=True, activation="tanh")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = autoencoder.fit(X, X, epochs=100, batch_size=64, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Plot training loss
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

# Load feature data for clustering
feature_data = pd.read_csv("processed_data/feature_data.csv")
true_labels = feature_data.iloc[:, 0].values  # Assuming first column is material labels
X_features = feature_data.iloc[:, 1:].values  # Exclude material labels

# Convert true labels to numerical format
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Ensure true_labels match reconstruction_error length for anomaly detection
true_labels_for_anomaly = true_labels[:len(reconstruction_error)]

# Compute per-material anomaly thresholds
thresholds = {}
for material in np.unique(true_labels_for_anomaly):
    material_mask = (true_labels_for_anomaly == material)
    material_errors = reconstruction_error[material_mask]
    thresholds[material] = np.percentile(material_errors, 95)

# Detect anomalies per material
anomalies = np.array([reconstruction_error[i] > thresholds[true_labels[i]] for i in range(len(reconstruction_error))])

# Plot reconstruction error
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label='Reconstruction Error')
for material, threshold in thresholds.items():
    plt.axvline(threshold, linestyle='dashed', label=f'{material} Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution per Material")
plt.legend()
plt.show()

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_features)

# Compute Silhouette Score
sil_score = silhouette_score(X_features, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Map clusters to true materials
cluster_to_material = {}
new_labels = np.zeros_like(cluster_labels)

for cluster_id in range(4):
    mask = (cluster_labels == cluster_id)
    if np.any(mask):  
        most_common_material = mode(true_labels_encoded[mask])[0][0]
        cluster_to_material[cluster_id] = most_common_material
        new_labels[mask] = most_common_material  

# Print cluster mappings
print("Cluster to Material Mapping:")
for cluster, material_index in cluster_to_material.items():
    material_name = label_encoder.inverse_transform([material_index])[0]
    print(f"Cluster {cluster} → Material: {material_name}")

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels_encoded, new_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Material")
plt.ylabel("True Material")
plt.title("Confusion Matrix for K-Means Clustering")
plt.show()

# Use t-SNE for visualization
X_tsne_3d = TSNE(n_components=3, perplexity=50, random_state=42).fit_transform(X_features)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=new_labels, cmap='viridis', alpha=0.7, edgecolors='k')
ax.set_xlabel("TSNE Component 1")
ax.set_ylabel("TSNE Component 2")
ax.set_zlabel("TSNE Component 3")
ax.set_title("3D t-SNE Visualization with Material-Aligned Labels")
cbar = plt.colorbar(scatter)
cbar.set_label("Material Label")
plt.show()
