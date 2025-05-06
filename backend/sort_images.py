import os
import cv2
import shutil

# Girdi klasörü (resimlerin bulunduğu yer)
input_dir = "cloth_dataset/tshirt"

# Hedef klasörler
short_dir = "cloth_dataset/short_sleeve"
long_dir = "cloth_dataset/long_sleeve"

# Hedef klasörleri oluştur (varsa sorun çıkarmaz)
os.makedirs(short_dir, exist_ok=True)
os.makedirs(long_dir, exist_ok=True)

# Resim dosyalarını sırala
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

for img_file in image_files:
    img_path = os.path.join(input_dir, img_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Cannot read {img_file}, skipping.")
        continue

    # Görüntüyü göster
    cv2.imshow("Image (press S for short sleeve, L for long sleeve)", img)
    key = cv2.waitKey(0)

    # Karar
    if key == ord('s'):
        shutil.move(img_path, os.path.join(short_dir, img_file))
        print(f"{img_file} => short_sleeve")
    elif key == ord('l'):
        shutil.move(img_path, os.path.join(long_dir, img_file))
        print(f"{img_file} => long_sleeve")
    elif key == 27:  # ESC tuşu
        print("Çıkılıyor...")
        break
    else:
        print("Geçersiz tuş, tekrar dene.")
        continue

cv2.destroyAllWindows()
