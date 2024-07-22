from fastapi import FastAPI, Request, HTTPException
import base64
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import gc
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  
    allow_methods=['*'],  
    allow_headers=['*']   
)

with open('bird_list_classes.json', 'r') as file:
    classes = json.load(file)

size = 299
img_size = (size, size)

# โหลด TFLite model
with open('model_quantized.tflite', 'rb') as f:
    tflite_model = f.read()

# แปลง TFLite model เป็น interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def base64_to_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def process_image(base64_input):
    try:
        img_array = base64_to_image(base64_input)

        # ทำการคาดคะเนด้วย TFLite model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Convert prediction to list of probabilities
        probabilities = list(predictions[0])

        # Thresholds
        threshold = 0.4
        isBird = True

        # Check if the maximum probability is below threshold
        if np.max(predictions) < threshold:
            isBird = False

        # Create a list of indexes sorted by probability
        sorted_indexes = np.argsort(probabilities)[::-1][:5]
        classes_id = []
        predict_rank = []
        probability_rank = []
        for i in sorted_indexes:
            id = str(i)
            class_name = classes.get(str(i), "Unknown")
            probability = probabilities[i]
            classes_id.append(id)
            probability_rank.append(probability.item())
            predict_rank.append(class_name)

        # Explicitly call garbage collection
        del img_array, predictions
        gc.collect()

        return {
            "predictions": [
                {"rank": 1, "class_id": classes_id[0], "class_name": predict_rank[0], "probabilities": probability_rank[0]},
                {"rank": 2, "class_id": classes_id[1], "class_name": predict_rank[1], "probabilities": probability_rank[1]},
                {"rank": 3, "class_id": classes_id[2], "class_name": predict_rank[2], "probabilities": probability_rank[2]},
                {"rank": 4, "class_id": classes_id[3], "class_name": predict_rank[3], "probabilities": probability_rank[3]},
                {"rank": 5, "class_id": classes_id[4], "class_name": predict_rank[4], "probabilities": probability_rank[4]}
            ],
            "isBird": isBird
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"Hello": classes}

@app.post("/api/birdClassify")
async def read_image(image_data: Request):
    try:
        image_data_json = await image_data.json()
        base64_input = image_data_json.get("image")
        if not base64_input:
            raise HTTPException(status_code=400, detail="No image data found")
        
        result = process_image(base64_input)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

