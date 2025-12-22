import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

classes = ['airplane', 'automobile', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Device:", device)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN()
model.load_state_dict(torch.load("simple_cifar10_cnn.pth", map_location=device))
model.to(device)
model.eval()

# ----- TRANSFORM -----
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

img_path = "dog.avif"   
img = Image.open(img_path).convert("RGB")
inp = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(inp)
    _, pred = output.max(1)
    label = classes[pred.item()]

print(f"Prediction: {label}")