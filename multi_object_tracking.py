from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Initialize tracker object and the tracking state
multiTracker = cv2.legacy.MultiTracker_create()
tracking = False

# Different colors for tracked objects
colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0)] 

# Helper function to initialize tracker
def initialize_tracker(frame, rois):
    global multiTracker, tracking
    multiTracker = cv2.legacy.MultiTracker_create() # Create multi-object tracker
    for roi in rois:
        tracker = cv2.legacy.TrackerCSRT_create() # Set the tracker for each ROI
        multiTracker.add(tracker, frame, tuple(roi)) # Add each tracker to the multiTracker
    tracking = True    

# Helper function to track multiple objects  
def track_object(frame):  
    global multiTracker, tracking
    if tracking:
        success, points = multiTracker.update(frame) # Update the tracker
        a = 0
        if success:
            for point in points:
                p1 = (int(point[0]), int(point[1]))
                p2 = (int(point[0] + point[2]), int(point[1] + point[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 3) # Draw rectangle around the tracked object
                cv2.rectangle(frame, p1, p2, colors[a], 3) # Use different colors for objects
                a += 1
                   
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
            rois = data["rois"] # ROI will be provided by the frontend
            initialize_tracker(frame, rois)
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
    app.run(host="0.0.0.0", port=5004)