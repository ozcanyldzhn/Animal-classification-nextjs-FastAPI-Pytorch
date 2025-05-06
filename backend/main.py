from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI()

# CORS ayarları (Nuxt frontend için gerekli)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşamasında * kullanılabilir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve sınıf adlarını yükle
class_names = ['Kısa Kollu', 'Uzun Kollu']

# Dummy model (eğitimli model gelene kadar)
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(224 * 224 * 3, 2)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

model = DummyModel()
model.eval()

# Eğer gerçek model geldiğinde şöyle yüklersin:
# model = torch.load('model/model.pt')
# model.eval()

# Görsel dönüştürme işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # (3, 224, 224)
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return {
        "label": class_names[predicted_idx.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
