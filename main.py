from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = FastAPI(title="Number Plate Recognition API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def preprocess_image(image):
    """Preprocess the image for better OCR recognition"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 170, 200)
    return edges


def extract_number_plate(image):
    """Extract number plate from the image"""
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx

    return None


@app.post("/recognize-plate")
async def recognize_number_plate(file: UploadFile = File(...)):
    """Endpoint to recognize number plate from uploaded image"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image = preprocess_image(image)
        plate_contour = extract_number_plate(processed_image)

        if plate_contour is None:
            raise HTTPException(status_code=400, detail="No number plate detected")

        x, y, w, h = cv2.boundingRect(plate_contour)
        number_plate_image = image[y:y + h, x:x + w]

        gray_plate = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)

        plate_text = pytesseract.image_to_string(
            gray_plate,
            config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()

        return {"number_plate": plate_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)