
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "customer_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load customer dataset
data = pd.read_csv(DATA_PATH)

# Select features for SOM
features = data[["Age", "AnnualIncome", "SpendingScore"]]

# Normalize data between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(features)

# Create and train the Self-Organizing Map
som = MiniSom(
    x=7,
    y=7,
    input_len=X.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

som.random_weights_init(X)
som.train_random(X, num_iteration=1000)

# Find the best matching SOM neuron for each customer
winners = np.array([som.winner(row) for row in X])

data["SOM_X"] = winners[:, 0]
data["SOM_Y"] = winners[:, 1]

# Save customer assignments
output_csv = os.path.join(RESULTS_DIR, "customer_clusters.csv")
data.to_csv(output_csv, index=False)

# Plot 1: SOM customer map
plt.figure(figsize=(8, 7))
plt.pcolor(som.distance_map().T, cmap="bone_r")
plt.colorbar(label="Relative neuron distance")

for i, (x, y) in enumerate(winners):
    plt.text(x + 0.5, y + 0.5, str(data.iloc[i]["CustomerID"]),
             ha="center", va="center", color="red", fontsize=8)

plt.title("SOM Customer Segmentation Map")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "som_map.png"), dpi=300)
plt.close()

# Plot 2: Customer visualization
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    data["AnnualIncome"],
    data["SpendingScore"],
    c=data["SOM_X"] * 7 + data["SOM_Y"],
    cmap="tab20",
    s=100
)

plt.colorbar(scatter, label="SOM neuron index")
plt.title("Customer Segmentation by Income and Spending")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "cluster_visualization.png"),
    dpi=300
)
plt.close()

print("SOM training completed successfully!")
print(f"Customers analyzed: {len(data)}")
print(f"Cluster assignments saved to: {output_csv}")
print("Visualizations saved in the results folder.")
