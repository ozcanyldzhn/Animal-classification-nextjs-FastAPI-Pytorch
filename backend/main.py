# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import io

# app = FastAPI()

# # CORS ayarları (Nuxt frontend için gerekli)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Geliştirme aşamasında * kullanılabilir
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Model ve sınıf adlarını yükle
# class_names = ['Kısa Kollu', 'Uzun Kollu']

# # Dummy model (eğitimli model gelene kadar)
# class DummyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = torch.nn.Flatten()
#         self.fc = torch.nn.Linear(224 * 224 * 3, 2)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.fc(x)

# model = DummyModel()
# model.eval()

# # Eğer gerçek model geldiğinde şöyle yüklersin:
# # model = torch.load('model/model.pt')
# # model.eval()

# # Görsel dönüştürme işlemi
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # (3, 224, 224)
# ])

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#         confidence, predicted_idx = torch.max(probabilities, dim=0)

#     return {
#         "label": class_names[predicted_idx.item()],
#         "confidence": round(confidence.item() * 100, 2)
#     }
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import io

# FastAPI uygulaması
app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model sınıfı
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli yükle
model = CNNModel()
model.load_state_dict(torch.load("model/cnn_model.pt", map_location=torch.device("cpu")))
model.eval()

class_names = ['long_sleeve', 'short_sleeve']

# Görsel ön işleme
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return JSONResponse(content={"prediction": predicted_class})

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}
