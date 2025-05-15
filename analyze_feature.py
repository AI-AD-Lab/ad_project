#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from dataset_single_entity import collate_fn_variable_length, LogDataset
from _utils.utils_path import sep_log_file, load_data
from model_structure import MultiVariableRNN, Simple1DCNN
# %%
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "test_scenario_logs"                  # From src
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

dataset = LogDataset(
    log_folder_path=SINGLE_SCENARIO_LOG_DATA, 
    simulation_folder='created_20250428150508', 
)

loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_variable_length)

# Training configuration
input_size = 7  # Number of features
num_classes = 9  # Number of output classes
num_epochs = 20
learning_rate = 0.0001

# Initialize model, loss function, and optimizer
model = MultiVariableRNN(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trained_model = Path('./model/best_model_20250430171456.pth')
if os.path.exists(trained_model):
    model.load_state_dict(torch.load(trained_model))
    print("Model loaded successfully.")
    model.eval()  # Set the model to evaluation mode
else:
    print("Model file not found. Training from scratch.")
    raise FileNotFoundError(f"Model file not found: {trained_model}")

output_list = []

total_label = []
total_predicted_label = []


for data, label in loader:
    # Zero the gradients
    optimizer.zero_grad()
    # Forward pass
    outputs = model(data)
    tmp_output = outputs.detach().numpy()

    softmax_output = F.softmax(outputs, dim=1).detach().numpy()
    tmp_output = softmax_output.reshape(-1, 9)
    tmp_output = np.argmax(tmp_output, axis=1)
    tmp_output = tmp_output.reshape(-1, 1)

    total_predicted_label.append(tmp_output)
    total_label.append(label.numpy())
#%%
total_label = np.array(total_label)
total_label = total_label.reshape(-1, 1)
print(total_label.shape)

total_predicted_label = np.array(total_predicted_label)
total_predicted_label = total_predicted_label.reshape(-1, 1)
print(total_predicted_label.shape)

#%%
accuracy = np.sum(np.array(total_predicted_label) == np.array(total_label)) / len(total_label)
print(f"Accuracy: {accuracy:.4f}")
# %%
output_list = np.array(output_list)
output_list = output_list.reshape(-1, 9)

output_predict_list = np.argmax(output_list, axis=1)
output_predict_list = output_predict_list.reshape(-1, 1)

print(output_predict_list)
# %%
for real, label in zip(total_label, output_predict_list):
    print(f"Probabilities: {real}, Predicted Label: {label}")

# %%
# dbscan 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances_argmin_min



