#%%
import torch
from dataset_single_entity import LogDataset, collate_fn_fixed_length, collate_fn_variable_length
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from datetime import datetime

from model_structure import MultiVariableRNN, Simple1DCNN

# %%

MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "test_scenario_logs"                  # From src
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

dataset = LogDataset(
    log_folder_path=SINGLE_SCENARIO_LOG_DATA, 
    simulation_folder='created_20250428150508', 
)
data_length = 5000
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


        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if total_loss < best_loss:
        best_loss = total_loss
        best_model = model.state_dict()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

data_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
best_model_path = Path('./model') / f"best_model_{data_time}.pth"
torch.save(best_model, best_model_path)
print(f"Best model saved to {best_model_path}")
print("Training complete.")