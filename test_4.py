# test_4在test_3的基础上的改进：
#   1. 重新调整batch size到32
#   2. 去除Early stopping，epoch由50增加到70

# 可以用来训练的版本2.0
# 训练结果分析：
#   1. 训练损失 下降得很顺利，从 0.84 下降到 0.79，说明模型仍在学习并收敛。
#   2. 验证损失 下降后在 50-60 轮附近开始波动，甚至略有上升，这可能意味着：
#               过拟合（模型对训练数据学得太好，但泛化能力下降）。
#   3. 误差分布在 0.39-0.44 之间，大部分数据点的误差集中在 0.40-0.42。
#      设定的 异常阈值在 ~0.45，比最高误差稍大，应该不会误判正常点。
#   4. 颜色代表 K-Means 分配的 4 个簇。
#      其中 0 号（紫色）簇占比远大于其他簇，说明簇间可能不均衡。
#      边界不太清晰，部分点可能被错误归类。

# 改进方案：test_5
#   1. 重新加上Early stopping，让训练在最优点提前停止，防止过拟合，避免训练损失下降但验证损失上升
#   2. 异常阈值调整，由mean + 3 * std改为95% 分位数 (np.percentile) 设定更稳定的阈值
#   3. 使用 t-SNE 进行降维可视化

# test_4 Improvements on test_3:
# 1. Resize batch size to 32.
# 2. Remove Early stopping and increase epoch from 50 to 70.

# Version 2.0 can be used for training.
# Analyze the training results:
# 1. The training loss decreased smoothly from 0.84 to 0.79, which means the model is still learning and converging.
# 2. The validation loss drops and then starts to fluctuate around rounds 50-60 and even rises slightly, which could mean:
# Overfitting (the model is learning too much about the training data, but generalization is decreasing).
# 3. Errors are distributed between 0.39 and 0.44, with most of the data points having errors centered around 0.40-0.42.
# The anomaly threshold is set at ~0.45, which is slightly larger than the highest error and should not misclassify normal points.
# 4. The colors represent the 4 clusters assigned by K-Means.
# 0 (purple color) is much larger than the other clusters, indicating that the clusters may not be balanced.
# The boundaries are not very clear and some points may be misclassified.


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
    "C:/Users/c1257/Desktop/processed_data/denoised_steel.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_roasted_steel.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_aluminum.csv",
    "C:/Users/c1257/Desktop/processed_data/denoised_brass.csv"
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
# 改进1&2
##########################################

# Train Autoencoder
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(X, X, epochs=70, batch_size=32, validation_split=0.1, verbose=1)

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
