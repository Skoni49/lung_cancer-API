from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import traceback

# ✅ Load models
lung_model = load_model("best_model_f.h5", compile=False)
filter_model = load_model("xray_filter_model.h5", compile=False)

# ✅ Class labels
class_labels = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Small Cell Carcinoma",
    "Squamous Cell Carcinoma",
]

# ✅ FastAPI app
app = FastAPI()

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://172.27.32.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Preprocess for both models
def preprocess_image(uploaded_file, size=(300, 300)) -> np.ndarray:
    img = Image.open(uploaded_file.file)
    img = img.convert("RGB")
    img = img.resize(size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ✅ Step 1: check if image is a valid chest X-ray
        filter_input = preprocess_image(file, size=(224, 224))
        is_valid_prob = filter_model.predict(filter_input)[0][0]

        if is_valid_prob < 0.6:  # threshold for "valid image"
            return JSONResponse(
                content={
                    "result_message": "❌ Invalid image. Please upload a valid chest X-ray."
                },
                status_code=200,
            )

        # ✅ Step 2: process for lung cancer classification
        file.file.seek(0)  # rewind file to read again
        lung_input = preprocess_image(file, size=(300, 300))
        predictions = lung_model.predict(lung_input)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        accuracy_percentage = round(confidence * 100, 2)

        label = class_labels[predicted_class]
        message = (
            "No, it's Normal."
            if label.lower() == "normal"
            else f"Yes, it's cancer '{label}'."
        )

        return JSONResponse(
            {"result_message": message, "accuracy": f"{accuracy_percentage}%"}
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
