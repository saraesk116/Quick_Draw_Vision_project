import torch
import torch.nn as nn
from tqdm import tqdm
from statistics import mean
from model import ConvNet
from dataset import QuickDrawMemmapDataset, build_default_class_files
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

# --- THIS IS THE TRAIN FUNCTION ---
# (We will add val_loader and writer later)
def train(net, optimizer, train_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(train_loader)
        for x, y in t:
            x, y = x.to(device), y.to(device) # Move data to device

            # TODO: Forward pass
            outputs = net(x)

            # TODO: Calculate loss
            loss = criterion(outputs, y)

            running_loss.append(loss.item())

            # TODO: Backward pass and optimization
            optimizer.zero_grad()
            loss.backward() # backward loss
            optimizer.step() # step of the optimizer

            t.set_description(f'training loss: {mean(running_loss)}')
    return running_loss

# --- THIS IS THE TEST FUNCTION SKELETON ---
def test(model, test_loader, device):
    model.eval() # Set model to evaluation mode
    test_corrects = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation
        for x, y in test_loader:
            # TODO: Move data to device
            x, y = x.to(device), y.to(device)

            # TODO: Get model predictions
            y_hat = model(x)
            #y_hat est un tensor de taille (batch_size, num_classes=10) contenant les scores pour chaque classe
            # TODO: Get the class with the highest score (argmax)
            predictions = torch.argmax(y_hat, dim=1)
            targets = torch.argmax(y, dim=1) if y.ndim > 1 else y

            # TODO: Count correct predictions
            test_corrects +=  (predictions == targets).sum().item()
            total += y.size(0)  # y et de taille (batch_size, num_classes=10)

    return test_corrects / total

# We will add the main execution block later
# if __name__ == '__main__':
#     ...