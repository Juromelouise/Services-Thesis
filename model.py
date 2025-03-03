import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import re
import os
from hezar.models import Model

# Load models
license_plate_detector = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")

# Load the TrOCR model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def detect_license_plate(image):
    plates = license_plate_detector(image)[0]
    plate_texts = []

    for plate in plates.boxes.data.tolist():
        x1, y1, x2, y2, _, _ = plate
        cropped_plate = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Apply Gaussian filter
        blurred_plate = cv2.GaussianBlur(cropped_plate, (5, 5), 0)
        
        # Convert to grayscale
        gray_plate = cv2.cvtColor(blurred_plate, cv2.COLOR_BGR2GRAY)
        
        # Apply additional preprocessing
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert the image to a PIL image
        pil_img = Image.fromarray(thresh_plate)

        # Enhance the image
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2)

        # Resize the image to improve OCR accuracy
        pil_img = pil_img.resize((pil_img.width * 4, pil_img.height * 4), Image.LANCZOS)

        # Convert the image to RGB
        rgb_pil_img = pil_img.convert("RGB")

        # Ensure the results directory exists
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Save the processed image to the results folder
        try:
            processed_image_path = os.path.join(results_dir, f'processed_plate_{x1}_{y1}.png')
            rgb_pil_img.save(processed_image_path)
        except Exception as e:
            print(f"Error saving image: {e}")

        # Preprocess the image for TrOCR
        pixel_values = processor(rgb_pil_img, return_tensors="pt").pixel_values

        # Generate the text using TrOCR
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Use regular expressions to filter the text and insert a space
        plate_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        plate_text = re.sub(r'([A-Z]{3})([0-9]{4})', r'\1 \2', plate_text)
        plate_texts.append(plate_text)

        # Calculate text size dynamically based on plate size
        plate_height = int(y2 - y1)
        text_scale = plate_height / 50  # Adjust divisor to fit your preference
        text_thickness = max(2, int(text_scale * 2))

        # Draw bounding box around plate
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), text_thickness)

        # Draw text (scaled)
        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
        text_x = int(x1)
        text_y = int(y1) - 10 if y1 - 10 > 0 else int(y1) + text_size[1] + 10

        cv2.putText(image, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness)

    return plate_texts, image