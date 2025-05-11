import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import os

# CUDA ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Model yapısını oluştur
model = models.efficientnet_b0(weights=None)
checkpoint = torch.load("model/animal_model.pt", map_location=device)
num_classes = len(checkpoint['class_names'])
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Veriyi hazırlama
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(image_path):
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} dosyası bulunamadı!")
        return None
        
    start_time = time.time()
    
    # Görüntüyü yükle ve işle
    image = Image.open(image_path).convert('RGB')
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
    print(f"Tahmin süresi: {inference_time*1000:.2f}ms")
    
    return results

if __name__ == "__main__":
    # Test edilecek görselin yolunu belirtin
    test_image = "backend/animals/cat/1.jpg"  # Düzeltilmiş görsel yolu
    
    # Görselin varlığını kontrol et
    if not os.path.exists(test_image):
        print(f"Hata: {test_image} dosyası bulunamadı!")
        print("\nLütfen geçerli bir test görseli yolu belirtin.")
        print("Örnek kullanım: python test_model.py")
        print("\nMevcut sınıflar:")
        for class_name in checkpoint['class_names']:
            print(f"- {class_name}")
        print("\nÖrnek görsel yolları:")
        print("- backend/animals/cat/1.jpg")
        print("- backend/animals/dog/1.jpg")
        print("- backend/animals/elephant/1.jpg")
    else:
        results = evaluate_model(test_image)
        if results:
            print("\nEn iyi 3 Tahmin:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['class']}: {result['confidence']}")
