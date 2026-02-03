import torch
import torch.nn as nn
from tqdm import tqdm
from statistics import mean
from model import ConvNet
from dataset import QuickDrawMemmapDataset, build_default_class_files, get_QuickDraw_dataloaders
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse

# --- THIS IS THE TRAIN FUNCTION ---
# (We will add val_loader and writer later)
def train(net, optimizer, train_loader, val_loader, exp_name, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    val_loss = []
 
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
        
        # Validation phase

        val_accuracy = test(net, val_loader, device)
        val_loss.append(val_accuracy)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_accuracy:.4f}')

        # save wieghts every epoch

        if not os.path.exists('weights'):
            os.makedirs('weights')
        if val_loss[-1] >= max(val_loss):
            torch.save(net.state_dict(), f'weights/{exp_name}_epoch{epoch+1}.pth')
            print(f'Saved model weights at epoch {epoch+1} with validation accuracy {val_accuracy:.4f}')
        
    return running_loss, val_loss

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
if __name__ == '__main__':
    # 1. Parse command-line arguments with argparse (see below)
    parser = argparse.ArgumentParser(description='Train a ConvNet on QuickDraw dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--exp_name', type=str, default='quickdraw_experiment', help='Experiment name for logging and saving (default: quickdraw_experiment)')
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    exp_name = args.exp_name
    print(f'Experiment: {exp_name}, Batch Size: {batch_size}, Learning Rate: {lr}, Epochs: {epochs}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Prepare dataset

    if not os.path.exists('runs'):
        os.makedirs('runs')
    writer = SummaryWriter(f'runs/{exp_name}')

    # 2. Load and preprocess data (sklearn train_test_split, TensorDataset)

    # 3. Create DataLoaders
    train_loader, val_loader, test_loader = get_QuickDraw_dataloaders(
        batch_size=batch_size,
        limit_per_class=None,
        num_workers=2      )
            
    # 4. Initialize model, optimizer with parsed args. I recommend to use AdamW.
    net = ConvNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    # 5. Call train() with train_loader and val_loader
    train_losses, val_accuracies = train(net, optimizer, train_loader, val_loader, exp_name, device, epochs=epochs)

    # 6. Test the model on test_loader and print accuracy
    test_accuracy = test(net, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    # 7. Save the model weights to weights/{exp_name}_net.pth
    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(net.state_dict(), f'weights/last_{exp_name}_net.pth')

    # 8. Add TensorBoard logging and embedding visualization here (No edit needed)
    #print("Adding embeddings to TensorBoard...")
    #8.a) Get 256 random images and labels from your train_dataset
    # perm = torch.randperm(len(train_dataset)) 
    # images, labels = train_dataset.tensors[0][perm][:256], train_dataset.tensors[1][perm][:256]
    # images = images.to(device)

    # # 8.b) Get embeddings from the model
    # with torch.no_grad():
    #     embeddings = net.get_features(images) # Use the method you defined!

    # # 8.c) Add to TensorBoard
    # writer.add_embedding(embeddings,
    #                     metadata=labels,
    #                     label_img=images.reshape(-1, 1, 28, 28), # Reshape for TB
    #                     global_step=1)

    # # 8.d) Save computational graph
    # writer.add_graph(net, images)

    # # 8.e) Save a sample of images
    # img_grid = torchvision.utils.make_grid(images.reshape(-1, 1, 28, 28)[:64])
    # writer.add_image('quickdraw_images', img_grid)

    # writer.close()
    # print("All done. Run 'tensorboard --logdir runs' to view.")