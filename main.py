from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import numpy as np
import cv2
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(debug=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to preprocess the image for OCR
def preprocess_image(image):
    try:
        # Resize image to normalize text size
        scale_percent = 300  # Upscale image by 300%
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive thresholding
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        return gray
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        raise

# Function to extract text from the preprocessed image
def detect_text_from_image(image, lang="eng"):
    try:
        # Preprocess the image for better OCR accuracy
        preprocessed_img = preprocess_image(image)
        # Configure Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # Ensure Tesseract is installed on Railway
        custom_config = r'--oem 3 --psm 7'  # OCR settings for a single line of text
        # Extract text
        text = pytesseract.image_to_string(preprocessed_img, config=custom_config, lang=lang)
        return text.strip()
    except Exception as e:
        logging.error(f"Error in detect_text_from_image: {e}")
        raise

# FastAPI endpoint for extracting number plate text
@app.post("/api/extract-number-plate")
async def extract_number_plate(file: UploadFile = File(...)):
    try:
        # Validate file size (max 10 MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"error": "File is too large. Maximum size is 10MB."}

        # Read the uploaded image file
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        # Extract text from the image
        extracted_text = detect_text_from_image(img_np)

        # If no text is detected, return an error message
        if not extracted_text:
            return {"error": "No text detected on the license plate."}

        return {"extractedText": extracted_text}

    except Exception as e:
        logging.error(f"Error in extract_number_plate: {e}")
        return {"error": "An error occurred while processing the image."}

# Run the application using: uvicorn main:app --reload
