from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Helper function to apply mosaic to a face region
def apply_mosaic(frame, x, y, w, h, level=15):
    mosaic = frame[y: y + h, x: x + w]
    mh = int(h / level)
    mw = int(w / level)
    mosaic = cv2.resize(mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y: y + h, x: x + w] = mosaic
    return frame

# API route to handle face mosaic
@app.route("/apply_mosaic", methods=["POST"])

def mosaic():
    try:
        # Decode the frame sent from the frontend
        data = request.json
        frame_str = data["frame"]
        frame_data = base64.b64decode(frame_str)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            frame = apply_mosaic(frame, x, y, w, h)

        # Encode the frame to send to the frontend
        _, buffer = cv2.imencode(".jpg", frame)
        frame_str = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"frame": frame_str})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500        

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

