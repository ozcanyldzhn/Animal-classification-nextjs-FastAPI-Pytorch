import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
from multiprocessing import freeze_support

def train():
    # CUDA kullanılabilirliğini kontrol et ve ayarla
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Veri yükleme ve ön işleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Veri kümesini yükle
    train_dataset = datasets.ImageFolder(root="animals", transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=0,  # Windows'ta multiprocessing sorunlarını önlemek için
        pin_memory=True
    )

    # EfficientNet-B0 modelini kullan
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Mixed precision training için scaler
    scaler = torch.amp.GradScaler()

    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    # Eğitim döngüsü
    epochs = 30
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")

        # En iyi modeli kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'class_names': train_dataset.classes,
                'accuracy': accuracy
            }, "model/animal_model.pt")
            print(f"New best model saved with accuracy: {accuracy:.2f}%")

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    print("Class names:", train_dataset.classes)

if __name__ == '__main__':
    freeze_support()  # Windows için multiprocessing desteği
    train()
