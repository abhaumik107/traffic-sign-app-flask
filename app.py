import os
import time
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# === Setup ===
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load models ===
print("📦 Loading CNN model...")
cnn_model = load_model(r"C:\Users\abhau\Downloads\tt100k_cnn_best (1).keras")

print("🧠 Loading YOLOv8 model...")
yolo_model = YOLO(r"C:\Users\abhau\Downloads\best_model_final\runs\detect\tt100k_yolo\weights\best.pt")

# === Class labels ===
custom_class_labels = [
    "Speed Limit 120", "Underpass Ahead", "Bus Only", "Kids Crossing", "Crossroad",
    "End of Speed Limit 40", "Fork Ahead", "H-Limit 4.5m", "H-Limit 4m", "H-Limit 5m",
    "Max Weight 20t", "Max Wt 30t", "Max Wt 55t", "Merge Right", "No Bicycles", "No Buses",
    "No Cars", "No Entry", "No Honking", "No Horn", "No Honking.", "No Left Turn", "No Right Turn",
    "No Stop", "No 2-Wheelers", "No U-Turn", "Pedestrian", "Pedestrian 1", "Pedestrian Walk",
    "Road Work", "Speed Limit 100", "Speed Limit 100.", "Speed Limit 20", "Speed Limit 30",
    "Speed Limit 40", "Speed Limit 5", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
    "Speed Limit 70.", "Speed Limit 80", "Speed Limit 80."
]

# === Initialize app ===
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <h2>🚦 Upload a Traffic Sign Image</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <input type="submit" value="Upload & Predict">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    # 🔄 Clean old files
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except:
            pass

    # 🖼️ Save uploaded image
    file = request.files['file']
    if file.filename == '':
        return "❌ No file uploaded"
    
    filename = f"input_{int(time.time())}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # === Step 1: YOLO detection ===
    image = cv2.imread(filepath)
    yolo_results = yolo_model(filepath, conf=0.1)
    boxes = yolo_results[0].boxes

    if boxes is None or len(boxes) == 0:
        return "⚠️ YOLO found no objects in the image."

    # === Step 2: Crop first detection box ===
    box = boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cropped = image[y1:y2, x1:x2]

    # 🐛 Check if crop is valid
    if cropped.size == 0:
        return "⚠️ Detected box was empty or invalid."

    crop_path = os.path.join(UPLOAD_FOLDER, "debug_crop.jpg")
    cv2.imwrite(crop_path, cropped)

    # === Step 3: CNN prediction ===
    resized = cv2.resize(cropped, (64, 64)) / 255.0
    input_img = np.expand_dims(resized, axis=0)
    prediction = cnn_model.predict(input_img)
    pred_class = int(np.argmax(prediction))
    pred_label = custom_class_labels[pred_class] if pred_class < len(custom_class_labels) else "Unknown"

    # === Step 4: Draw bounding box and label ===
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, pred_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_name = "result.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, result_name)
    cv2.imwrite(result_path, image)

    # === Step 5: Return result in HTML ===
    return render_template_string(f'''
        <h2>✅ YOLO + CNN Prediction</h2>
        <p><b>Predicted class:</b> {pred_label}</p>
        <h3>📷 Full Image with Detection:</h3>
        <img src="/uploads/{result_name}" width="800"><br><br>
        <h3>🔍 Cropped Region Used for CNN:</h3>
        <img src="/uploads/debug_crop.jpg" width="200"><br><br>
        <a href="/">🔁 Try Another</a>
    ''')

# === Static image route ===
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# === Run app ===
if __name__ == '__main__':
    print("✅ Flask app starting...")
    app.run(debug=True, port=5050)

