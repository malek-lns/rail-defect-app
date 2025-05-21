import cv2
import numpy as np
import os
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("models/rail_detector.pt")  

def detect_and_crop_multiple(image_path, target_size=(128, 128), stride=64, zoom_factor=1.2, output_folder="crops"):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Prédiction avec YOLO
    results = model.predict(source=image_rgb, conf=0.1, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("❌ Aucun rail détecté.")
        return []

    # Box englobante de toutes les boxes détectées
    x1 = max(0, min([int(b.xyxy[0][0]) for b in boxes]))
    y1 = max(0, min([int(b.xyxy[0][1]) for b in boxes]))
    x2 = min(w, max([int(b.xyxy[0][2]) for b in boxes]))
    y2 = min(h, max([int(b.xyxy[0][3]) for b in boxes]))

    # Appliquer un zoom centré
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    box_w = x2 - x1
    box_h = y2 - y1

    zoom_w = int(box_w * zoom_factor / 2)
    zoom_h = int(box_h * zoom_factor / 2)

    x1 = max(0, cx - zoom_w)
    y1 = max(0, cy - zoom_h)
    x2 = min(w, cx + zoom_w)
    y2 = min(h, cy + zoom_h)

    # Découpage
    roi = image_rgb[y1:y2, x1:x2]
    roi_h, roi_w, _ = roi.shape

    if roi_h < target_size[1] or roi_w < target_size[0]:
        print(f"⚠️ Zone trop petite ({roi_w}x{roi_h}) pour des crops {target_size}")
        return []

    os.makedirs(output_folder, exist_ok=True)
    crops = []
    count = 0
    for y in range(0, roi_h - target_size[1] + 1, stride):
        for x in range(0, roi_w - target_size[0] + 1, stride):
            crop = roi[y:y+target_size[1], x:x+target_size[0]]
            crop_path = os.path.join(output_folder, f"crop_{count}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            crops.append(crop_path)
            count += 1

    print(f"✅ {count} crops enregistrés dans {output_folder}")
    return crops
