import PIL
from fastapi import FastAPI, UploadFile, HTTPException, File
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request
import numpy as np
from PIL import Image
import io
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# Load the model
model = tf.keras.models.load_model("kanji_model.keras")


def preprocess_image(image_data: bytes) -> tf.Tensor:
    # Convert the byte stream to a PIL Image
    try:
        image = Image.open(io.BytesIO(image_data))
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=400,
                            detail="Unable to read the uploaded image. Please ensure it's a valid image and try again.")
    # Resize the image to the required input size for your model (e.g., 128x128)
    image = image.resize((256, 256))
    # Convert the PIL image to a numpy array and normalize
    image_array = np.asarray(image) / 255.0
    # Ensure the image is in the shape expected by your model
    # If your model expects a 4D tensor (batch_size, height, width, channels),
    # you'll need to expand dimensions
    tensor = tf.expand_dims(image_array, 0)
    return tensor


def postprocess_prediction(prediction: np.ndarray) -> dict:
    predicted_value = prediction[0][0]  # If prediction is already a numpy array, directly extract the value
    # If the prediction is closer to 1, it's "GOOD", otherwise "BAD"
    # Using 0.5 as the threshold
    result = "Bueno" if predicted_value > 0.5 else "Malo"
    percentage_score = "{:.2f}%".format(float(predicted_value) * 100)
    return {"RESULT": result, "SCORE": percentage_score}


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Hello World!"}


@app.get("/upload")
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="File not provided")
    allowed_content_types = ["image/jpeg", "image/png", "image/gif", "image/webp", ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image.")
    try:
        image_data = await file.read()  # Convert uploaded file to bytes
        tensor = preprocess_image(image_data)  # Preprocess image data
        prediction = model.predict(tensor)  # Make prediction
        prediction_result = postprocess_prediction(
            prediction)  # Convert prediction to human-readable format ("GOOD" or "BAD")
        return JSONResponse(content=prediction_result)
    except Exception as e:
        return JSONResponse(content={"error reading the file": str(e)}, status_code=400)
