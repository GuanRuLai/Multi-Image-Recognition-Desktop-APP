from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Helper function to detect features (eyes, nose, mouth)
def detect_features(frame):
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    gray = cv2.medianBlur(frame, 1) # Remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) # Further noise removal

    # Eye detection
    eyes = eye_cascade.detectMultiScale(gray)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green for eyes

    # Mouth detection
    mouths = mouth_cascade.detectMultiScale(gray)
    for (x, y, w, h) in mouths:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red for mouth

    # Nose detection
    noses = nose_cascade.detectMultiScale(gray)
    for (x, y, w, h) in noses:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue for nose    

    return frame

# API route to handle feature detection
@app.route("/detect_features", methods=["POST"])

def detect():
    try:
        # Decode the frame sent from the frontend
        data = request.json
        frame_str = data["frame"]
        frame_data = base64.b64decode(frame_str)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
        frame = detect_features(frame)

        # Encode the frame to send back to the frontend
        _, buffer = cv2.imencode(".jpg", frame)
        frame_str = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"frame": frame_str})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500       

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)    