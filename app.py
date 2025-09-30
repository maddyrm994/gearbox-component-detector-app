import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'assembly_model.keras'
CLASS_NAMES = {0: 'Additional Component', 1: 'Correct Assembly', 2: 'Missing Component'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Model ---
# Load the Keras model once when the application starts.
try:
    print("Loading Keras model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load model from {MODEL_PATH}. Error: {e}")
    model = None

# --- Helper Function to Preprocess Image ---
# This is a direct copy of the function from your Streamlit app.
def preprocess_image(img_array):
    # Expects a NumPy array in RGB format
    resized_img = cv2.resize(img_array, (224, 224))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=0)

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/inspect_image', methods=['POST'])
def inspect_image():
    """Handles the image upload and returns a prediction."""
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read image, convert to RGB, and process
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(img_rgb)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Unknown")

        return jsonify({
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.2f}"
        })

    return jsonify({"error": "An error occurred"}), 500

def generate_frames():
    """A generator function that yields camera frames for streaming."""
    if model is None:
        print("Cannot start camera stream, model not loaded.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = preprocess_image(frame_rgb)
            
            # Predict
            prediction = model.predict(processed_frame)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Unknown")

            # Determine color and text for the overlay
            status_text = f"Status: {predicted_class_name} ({confidence:.2f}%)"
            color = (0, 0, 0) # Black (default)
            if predicted_class_name == 'Correct Assembly':
                color = (0, 255, 0) # Green
            elif predicted_class_name == 'Missing Component':
                color = (0, 0, 255) # Red (Note: OpenCV uses BGR)
            elif predicted_class_name == 'Additional Component':
                color = (0, 255, 255) # Yellow

            # Draw the text on the original BGR frame
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)