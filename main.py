from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from utils.yolo_crop import detect_and_crop_multiple
from utils.classifier import predict_defauts

app = FastAPI()

# Config templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "uploads"
CROPS_FOLDER = "crops"

# Création des dossiers si pas déjà présents
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload de l'image
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"✅ Image sauvegardée : {file_path}")
    return RedirectResponse(url="/", status_code=303)



@app.post("/detect-rail")
async def detect_rail():
    # Récupérer l’image uploadée la plus récente
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    if not uploaded_files:
        return {"error": "Aucune image uploadée."}

    latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in uploaded_files], key=os.path.getctime)
    print(f"🖼️ Traitement de : {latest_file}")

    # Supprimer les anciens crops
    for f in os.listdir(CROPS_FOLDER):
        os.remove(os.path.join(CROPS_FOLDER, f))

    # Appliquer le modèle de détection et générer les crops
    crops = detect_and_crop_multiple(latest_file, output_folder=CROPS_FOLDER)

    return RedirectResponse(url="/", status_code=303)



@app.post("/detect-defauts")
async def detect_defauts():
    results = predict_defauts()

    # Stocker les résultats dans un fichier temporaire (optionnel)
    with open("crops/results.txt", "w") as f:
        for filename, label in results:
            f.write(f"{filename}: {label}\n")

    return RedirectResponse(url="/resultats", status_code=303)


