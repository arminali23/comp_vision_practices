import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('device: ', device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)   
print("Batch labels shape:", labels.shape)

classes = train_dataset.classes
print("Num classes:", len(classes))
print("Classes:", classes)