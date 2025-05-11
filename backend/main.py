from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import io
import time

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

# CUDA ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Model yapısını oluştur
model = models.efficientnet_b0(weights=None)
checkpoint = torch.load("model/animal_model.pt", map_location=device)
num_classes = len(checkpoint['class_names'])
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Görsel ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Görüntüyü oku ve işle
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Tahmin yap
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    # Sonuçları hazırla
    results = []
    for prob, idx in zip(top3_prob, top3_indices):
        results.append({
            "class": checkpoint['class_names'][idx.item()],
            "confidence": f"{prob.item() * 100:.2f}%"
        })
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f}ms")
    
    return JSONResponse(content={
        "predictions": results,
        "inference_time_ms": f"{inference_time*1000:.2f}"
    })

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}
