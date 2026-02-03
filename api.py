from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import torch
import io
from PIL import Image
from typing import List

# Import your model architecture
# Ensure model.py is in the same directory or properly in PYTHONPATH
from model import ConvNet
import torchvision.transforms as transforms


# Hardcoded classes (Must match the order of your training!)
CLASS_NAMES = ["airplane","banana","cat","alarm clock","dolphin","circle","door",
               "eye","moon","donut"]


# Global variables to store the model and device
model = None
device = None

# --- LIFESPAN (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function executes when the API starts.
    Use it to load the model into memory.
    """
    global model, device

    # TODO: Set the device (cuda or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Initialize the ConvNet model
    model = ConvNet().to(device)

    # TODO: Load the trained weights
    # Hint: Use torch.load() and model.load_state_dict()
    # Make sure to map_location=device if you trained on GPU but run API on CPU
    model_path = "weights.pth" # Update this path!
    print(f"Loading model from {model_path}...")

    # ... load state dict ...
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # TODO: Set model to eval mode
    model.eval()



    print("Model loaded successfully!")
    yield
    # Code here would run on shutdown (not needed for now)
    print("Shutting down...")

# --- API INITIALIZATION ---
app = FastAPI(title="Quick, Draw! API", lifespan=lifespan)

# --- PREPROCESSING ---
def transform_image(image_bytes):
    """
    Transforms raw image bytes into a PyTorch tensor.
    Steps:
    1. Open bytes with PIL
    2. Convert to Grayscale (L)
    3. Resize to 28x28
    4. Convert to Tensor and Normalize
    """
    image = Image.open(io.BytesIO(image_bytes))

    # TODO: Convert to grayscale
    image = image.convert("L")

    # TODO: Resize to 28x28
    image = image.resize((28, 28))

    # TODO: Convert to numpy/tensor, normalize to [0, 1], and add batch dimension
    # The final shape must be (1, 1, 28, 28)
    tensor = transforms.ToTensor()(image).unsqueeze(0)

    return tensor.to(device)

# --- ENDPOINTS ---

@app.get("/")
def index():
    return {"message": "Welcome to the Quick, Draw! API. Use /predict to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a single image file, processes it, and returns the prediction.
    """
    # TODO: Read the file content
    contents = await file.read()

    # TODO: Transform the image using your helper function
    input_tensor = transform_image(contents)

    # TODO: Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, prediction_idx].item()

    # TODO: Return JSON response
    return {
        "filename": file.filename,
        "class": CLASS_NAMES[prediction_idx],
        "confidence": confidence
    }

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Receives a list of image files and returns a list of predictions.
    """
    results = []

    # TODO: Loop through the files
    for file in files:
        contents = await file.read()
        input_tensor = transform_image(contents)
        with torch.no_grad():
            output = model(input_tensor)
            prediction_idx = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, prediction_idx].item()
        results.append({
            "filename": file.filename,
            "class": CLASS_NAMES[prediction_idx],
            "confidence": confidence
        })

    return {"results": results}