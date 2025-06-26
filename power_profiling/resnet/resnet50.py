import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from imagenetv2_pytorch import ImageNetV2Dataset
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='ResNet50 training on ImageNetV2')
parser.add_argument('--n', '-n', type=int, default=256, help='Batch size')
parser.add_argument('--i', '-i', type=int, default=50, help='Number of iterations')
args = parser.parse_args()

# Set up transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = ImageNetV2Dataset(
    "matched-frequency", transform=transform)
dataloader = DataLoader(dataset, batch_size=args.n, shuffle=True)

# Load model
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 1000)
resnet50 = resnet50.to('cuda:0')

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)

# Set number of iterations to run from command line argument
running_iters = args.i

# Set model to training mode
resnet50.train()

# Run only the specified number of batches
for i, (inputs, labels) in enumerate(dataloader):
    # Move data to GPU
    inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
    
    # Forward pass
    optimizer.zero_grad()
    outputs = resnet50(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print loss for each batch
    print(f'Batch {i+1}/{running_iters}, Loss: {loss.item():.4f}')
    
    # Stop after reaching specified number of iterations
    if i + 1 >= running_iters:
        break

print("Testing completed!")