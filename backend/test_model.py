import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from PIL import Image

# CNNModel sınıfını burada tanımlıyoruz
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)  # İki sınıf: kısa kollu ve uzun kollu

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli yükle
model = CNNModel()
model.load_state_dict(torch.load("model/cnn_model.pt"))
model.eval()

# Veriyi hazırlama (aynı dönüşümler)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test verisi ile modeli değerlendirme
# Model tahminlerini düzeltme
def evaluate_model(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Modelin beklediği forma sokma

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Model çıktısını düzeltme
    if predicted.item() == 0:
        return "Long Sleeve"
    else:
        return "Short Sleeve"


# Test et
test_image = "cloth_dataset/long_sleeve/2.jpg"  # Test etmek istediğiniz görselin yolu
print(f"Predicted: {evaluate_model(test_image)}")
