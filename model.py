import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, dropout_rate=0.5 ):
        super(ConvNet, self).__init__()
        # TODO: Define all your layers here
        self.conv1 = nn.Conv2d(in_channels= 1,out_channels = 16 , kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels= 16,out_channels = 32 , kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.Lin1 = nn.Linear(in_features=32 * 5 * 5, out_features=128)
        self.Lin2 = nn.Linear(in_features=128, out_features=10)
        # par defaut padding = 0 , stride = 1
    def forward(self, x):
        # TODO: Define the forward pass
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # Flatten tensor for the linear layer
        x = F.relu(self.Lin1(x))
        x = self.Lin2(x)
    
        return x

    def get_features(self, x):
        # This method will be used for TensorBoard embeddings
        # It should return the flattened output of the last conv layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x