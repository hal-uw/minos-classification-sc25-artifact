import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

# 下载和解压数据集

# 定义自定义数据集类
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.Resize(256),  # 调整大小以符合ResNet50的输入
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageNetV2Dataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)


dataset = ImageNetV2Dataset(
    "imagenetv2-matched-frequency-format-val", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 1000)  # 调整输出层以匹配ImageNet的类别数
resnet50 = resnet50.to('cuda:0')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)
warmup_iters = 5
running_iters = 50

resnet50.train()

for i, (inputs, labels) in enumerate(dataloader):
    if i > running_iters:
        break
    if i == warmup_iters:
        print("start profiling")
        torch.cuda.cudart().cudaProfilerStart()
    if i >= warmup_iters:
        nvtx.range_push(f'iteration_{i}')
    inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
    optimizer.zero_grad()
    outputs = resnet50(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if i >= warmup_iters:
        nvtx.range_pop()
    print(f"iteration {i} loss: {loss.item()}")
print("stop profiling")
torch.cuda.cudart().cudaProfilerStop()
