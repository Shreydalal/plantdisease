from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
from pathlib import Path
from keras.layers import TFSMLayer  # type: ignore
import tensorflow as tf
from fastapi.responses import JSONResponse

# Initialize the app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the "client" folder to serve static files (CSS, JS, images)
CLIENT_FOLDER = Path("client")
app.mount("/client", StaticFiles(directory=CLIENT_FOLDER), name="client")

# Serve the homepage
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    index_file = CLIENT_FOLDER / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    else:
        return HTMLResponse(content="index.html not found", status_code=404)


# Model configuration
MODEL_PATH = r"C:\Users\User\Desktop\Deep Learning\models\model_1"
MODEL = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Function to read and process the uploaded image
def read_file_as_image(data: bytes) -> np.ndarray:
    """Converts uploaded file bytes into a numpy array image."""
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3-channel image
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image).astype("float32")  # Convert to float32
    image /= 255.0  # Normalize pixel values to [0, 1]
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image_data = await file.read()
        image = read_file_as_image(image_data)
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = MODEL(img_batch)
        predictions_tensor = predictions.get('dense_1', None)
        if predictions_tensor is None:
            raise ValueError("Key 'dense_1' not found in model output.")

        predictions = predictions_tensor.numpy()
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        return JSONResponse(content={
            "class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the server
#if __name__ == "__main__":
 #   uvicorn.run(app, host="127.0.0.1", port=8000)
