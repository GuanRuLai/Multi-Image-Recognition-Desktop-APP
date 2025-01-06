from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Initialize tracking state 
tracking = False

# Helper function to initialize tracker
def initialize_tracker(frame, roi):
    global tracker, tracking
    tracker = cv2.TrackerCSRT_create() # create the tracker
    tracker.init(frame, roi)
    tracking = True

# Helper function to track object  
def track_object(frame):  
    global tracker, tracking
    if tracking:
        success, point = tracker.update(frame) # Update the tracker
        if success:
            p1 = (int(point[0]), int(point[1]))
            p2 = (int(point[0] + point[2]), int(point[1] + point[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 3) # Draw rectangle around the tracked object
    
    return frame

# API route to handle object tracking
@app.route("/object_tracking", methods=["POST"])

def object_tracking():
    try:
        # Decode the frame sent from the frontend
        data = request.json
        frame_str = data["frame"]
        frame_data = base64.b64decode(frame_str)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get the mode (default to "track")
        mode = data.get("mode", "track") 

        if mode == "initialize":
            roi = data["roi"] # ROI will be provided by the frontend
            roi_tuple = tuple(map(int, roi)) # Convert ROI to tuple (x, y, w, h)
            initialize_tracker(frame, roi_tuple)
            return jsonify({"status": "tracking initialized"})   
        
        elif mode == "track":
            processed_frame = track_object(frame)   

            # Encode the frame to send to the frontend
            _, buffer = cv2.imencode(".jpg", processed_frame)
            frame_str = base64.b64encode(buffer).decode("utf-8")

            return jsonify({"frame": frame_str})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
           
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)