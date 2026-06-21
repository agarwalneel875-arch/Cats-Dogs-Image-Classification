import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "cats_dogs_resnet18.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded successfully")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

print("Loading image...")

image = Image.open("test.jpg").convert("RGB")

image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)

classes = ["cats", "dogs"]

print("Prediction:", classes[pred.item()])