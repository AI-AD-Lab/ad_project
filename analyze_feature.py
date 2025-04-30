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

best_loss = float('inf')
best_model = None

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for data, label in loader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        
        log_probs = F.log_softmax(outputs, dim=1)
        targets = F.one_hot(label, num_classes=outputs.size(1)).float()
        loss = criterion(log_probs, targets)


