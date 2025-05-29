from torch import nn
import torch.nn.functional as F


class Simple1DCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=8, kernel_size=5, stride=2, padding=2)  # Output: 8 x 2500
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2)  # Output: 8 x 1250
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2)  # Output: 8 x 625
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2)  # Output: 8 x 312

        self.fc1 = nn.Linear(8 * 313, 8*312)
        self.fc2 = nn.Linear(8*312, 4*312)  # Second fully connected layer
        self.fc3 = nn.Linear(4*312, 312)
        self.fc4 = nn.Linear(312, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, num_classes)  # Third fully connected layer
        

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc6(x)
        # x  = x.softmax(dim=1)  # Apply softmax to the output
        return x


class MultiVariableRNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiVariableRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, 64, batch_first=True)  # LSTM layer
        self.fc1 = nn.Linear(64, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, num_classes)  # Second fully connected layer

    def forward(self, x):
        x, _ = self.rnn(x)  # Pass through LSTM layer
        x = x[:, -1, :]  # Get the last time step output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x