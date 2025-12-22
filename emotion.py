import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('device: ', device)

data_dir = './DATASET'
train_dir = os.path.join(data_dir,'train')
val_dir = os.path.join(data_dir,'test')



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ImageFolder(root = train_dir, transform=transform)
test_dataset = ImageFolder(root = val_dir, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)
print("Num classes:", num_classes)

class simpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
        
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = simpleCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-3)

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"[Train] Epoch {epoch} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")


def evaluate(epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"[Val]   Epoch {epoch} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")
    return epoch_acc

best_acc = 0.0
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch)
    val_acc = evaluate(epoch)

    # en iyi modeli kaydet
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "emotion_cnn_best.pth")
        print(f"  >> New best model saved (acc={best_acc:.4f})")

print("Training finished. Best val acc:", best_acc)