from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from numpy import linalg as LA
import subprocess
import sys

# Ensure the "keras-facenet" module is installed
try:
    from keras_facenet import FaceNet
except ImportError:
    print("keras_facenet isn't installed, installed...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-facenet"])
    from keras_facenet import FaceNet

app = Flask(__name__)

# Initialize the pre-trained FaceNet model
embedder = FaceNet()

# Helper function to preprocess images and generate face embeddings
def get_face_embeddings(frame):
    # Detect faces and extract embeddings from the images
    results = embedder.extract(frame, threshold=0.95)
    if len(results) == 0:
        raise ValueError(f"No face detected.")
    
    # Get the embedding vector
    embeddings = results[0]["embedding"]
    # Get bounding box to display on the frame
    bbox = results[0]["box"]  

    return embeddings, bbox

# Helper function to compare two face embeddings
def compare_faces(embedding1, embedding2, threshold=0.7):
    # Calculate Euclidean distance between two face embeddings
    distance = LA.norm(embedding1 - embedding2)

    # Check if the distance < threshold (indicating a match)
    matched = distance < threshold

    return matched, distance 

# API route to handle real-time face detection and matching
@app.route("/detect_face", methods=["POST"])

def detect_face():
    try:
        # Decode the frame sent from the frontend
        data = request.json
        frame_str = data["frame"]
        reference_embedding = data.get("reference_embedding", None)
        frame_data = base64.b64decode(frame_str)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        embeddings,_ = get_face_embeddings(frame)
        if embeddings is None:
            return jsonify({"message": "No face detected."}), 400
        
        if reference_embedding is None:
            # Set the first detected face as the reference
            reference_embedding = embeddings.tolist() # Convert to list to make it JSON serializable
            response = {
                "message": "Reference face set.",
                "reference_embedding": reference_embedding
            }
        else:
            # Compare the detected face with the reference
            reference_embedding = np.array(reference_embedding) # Convert back to numpy array
            matched, distance = compare_faces(embeddings, reference_embedding)

            if matched:
                message = "Faces matched!"
            else:
                message = "Faces not matched!"           
            response = {
                "message": message,
                "distance": distance
            }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500   
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)