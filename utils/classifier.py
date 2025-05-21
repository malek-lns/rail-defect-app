from ultralytics import YOLO
import os

# Charger ton modèle YOLOv8 de classification/détection de défauts
model = YOLO("models/default_classifier.pt")  # ← ton deuxième modèle YOLO

classes = ["healthy", "joint", "squat", "ssquat"]

def predict_defauts(crops_folder="crops"):
    results = []

    for filename in os.listdir(crops_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(crops_folder, filename)

            pred = model.predict(source=path, conf=0.25, verbose=False)
            boxes = pred[0].boxes

            if boxes is None or len(boxes) == 0:
                label = "healthy"
            else:
                # Si plusieurs défauts détectés → garder celui avec la plus haute confiance
                best = boxes.conf.argmax().item()
                class_id = int(boxes.cls[best].item())
                label = classes[class_id]

            results.append((filename, label))

    print(f"✅ Prédictions YOLOv8 terminées sur {len(results)} crops")
    return results
