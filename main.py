import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from io import BytesIO
from PIL import Image
from keras.layers import TFSMLayer  # type: ignore
import tensorflow as tf
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

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
MODEL_PATH = "./model"
MODEL = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Function to read and process the uploaded image
def read_file_as_image(data: bytes) -> np.ndarray:
    """Converts uploaded file bytes into a numpy array image."""
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3-channel image
        image = image.resize((256, 256))  # Resize to match model input size
        image = np.array(image).astype("float32")  # Convert to float32
        image /= 255.0  # Normalize pixel values to [0, 1]
        return image
    except Exception as e:
        logging.error(f"Error in image processing: {e}")
        raise ValueError("Error processing image.")

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Create a temporary directory and save the uploaded image
        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        temp_file_path = os.path.join(temp_dir, file.filename)  # Define the file path

        # Save the uploaded image to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Log the path of the saved file (optional, for debugging purposes)
        logging.debug(f"File saved temporarily at: {temp_file_path}")

        # Read and process the uploaded image
        image = read_file_as_image(open(temp_file_path, "rb").read())
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = MODEL(img_batch)
        predictions_tensor = predictions.get('dense_1', None)
        if predictions_tensor is None:
            raise ValueError("Key 'dense_1' not found in model output.")

        predictions = predictions_tensor.numpy()
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        # Convert np.float32 to float to make it JSON serializable
        confidence = float(confidence)  # Convert to native Python float

        # Return the prediction and the temporary file path
        return JSONResponse(content={
            "class": predicted_class,
            "confidence": confidence,  # Confidence is now a native float
            "file_path": temp_file_path  # Include the file path in the response
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: Clean-up the temporary file after processing (if necessary)
# This function can be added after the model's response is returned.
# def cleanup_temp_file(file_path: str):
#     try:
#         os.remove(file_path)
#         logging.info(f"Temporary file {file_path} removed successfully.")
#     except Exception as e:
#         logging.error(f"Error removing temporary file {file_path}: {e}")

# Run the server (commented out for deployment)
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
