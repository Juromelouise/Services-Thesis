from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from model import detect_license_plate

router = APIRouter()

@router.post("/detect_plate/")
async def detect_plate(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        plate_texts, processed_image = detect_license_plate(image)
        
        _, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        
        return JSONResponse(content={"license_plate": plate_texts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))