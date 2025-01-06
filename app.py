import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import base64
import requests
import numpy as np
import subprocess
import time
import threading

# Define functions to activate the Flask server(customized for each function)
def start_flask_server(script_name, server_name, wait_time=3):
    def run_backend():
        subprocess.Popen(["python", script_name])

    backend_thread = threading.Thread(target=run_backend)
    backend_thread.start()   

    # Try to connect to a server until the server starts successfully
    server_started = False
    for _ in range(wait_time): # Wait for the specified time to check every second
        try:
            response = requests.get(server_name)
            if response.status_code == 200:
                print("Server started successfully")
                server_started = True
                break        
        except requests.ConnectionError:
            print("Waiting for server to start...")
            time.sleep(1)

    if not server_started:
        print(f"Failed to start the server after {wait_time} seconds")

# Initialize the main application window
root = tk.Tk()
root.title("Image Recognition Application")
root.geometry("640x640")
root.configure(bg="#d0d0d0") # light gray background 

# Create a frame to center content vertically
content_frame = tk.Frame(root, bg="#d0d0d0")
content_frame.pack(expand=True)  # Expand the frame to fill the window and allow vertical centering

# Title design
title_label = ttk.Label(
    content_frame, 
    text="Image Recognition Application", 
    font=("Helvetica", 20, "bold"), 
    background="#d0d0d0", 
    foreground="#333"
)
title_label.pack(pady=20)

# Button floating effect（hover state）
def on_enter(e):
    e.widget["background"] = "#add8e6" # The button turns light blue when suspended
    e.widget["relief"] = "raised" # Add shadow effect
def on_leave(e, bg_color):
    e.widget["background"] = bg_color # Restore the specified background color
    e.widget["relief"] = "flat"

# Universal button style
def create_button(text, command, bg_color):
    button = tk.Button(
        content_frame, 
        text=text, 
        width=30, 
        height=2, 
        bg=bg_color, 
        fg="#333", 
        font=("Helvetica", 14, "bold"),
        relief="flat", # Borderless, modern flat design 
        activebackground="#add8e6", # Effect when pressed
        cursor="hand2" # Make the mouse into hand shape 
    )
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", lambda e: on_leave(e, bg_color))
    button.config(command=command)
    return button


# """Define the operation of each function"""

# Function1: face detection
def face_detection():
    # Activate the backend file
    start_flask_server("face_detection.py", "http://127.0.0.1:5000", wait_time=5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a new window to display the video feed
    detection_window = tk.Toplevel(root)
    detection_window.title("Face Detection")
    lmain = tk.Label(detection_window)
    lmain.pack()

    # Create a label to display the matching result
    result_label = tk.Label(detection_window, text="Matching Result: ", font=("Helvetica", 14))
    result_label.pack()

    # Initialize the frame counter & reference embedding
    frame_counter = 0  
    reference_embedding = None

    def show_frame():
        nonlocal frame_counter
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return
        
        # Reduce size to speed up
        frame = cv2.resize(frame, (640, 360)) 

        # Send the frame to the server for processing every 10 frames
        if frame_counter % 10 == 0:
            # Encode the frame to send to the server
            _, buffer = cv2.imencode(".jpg", frame) 
            frame_str = base64.b64encode(buffer).decode("utf-8")

            try:
                # Send the frame to the backend for face detection and comparison
                data_to_send = {"frame": frame_str}
                nonlocal reference_embedding
                if reference_embedding:
                    data_to_send["reference_embedding"] = reference_embedding

                response = requests.post("http://127.0.0.1:5000/detect_face", json=data_to_send)
                data = response.json()

                # Check if a reference face was set or if comparison results were returned
                if "reference_embedding" in data:
                    reference_embedding = data["reference_embedding"]
                    result_label.config(text="Reference face set.")
                else:
                    result_label.config(text=f"{data['message']} (Distance: {data['distance']:.2f})")      

            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.content}")
                return
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return

        # Convert frame to image and display in Tkinter window
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    show_frame() # Start displaying the video feed

    detection_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), detection_window.destroy()))

# Function 2: face mosaic   
def face_mosaic():
    # Activate the backend file
    start_flask_server("face_mosaic.py", "http://127.0.0.1:5001")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a new window to display the video feed
    mosaic_window = tk.Toplevel(root)
    mosaic_window.title("Face Mosaic")
    lmain = tk.Label(mosaic_window)
    lmain.pack()

    def show_frame():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return

        # Reduce size to speed up
        frame = cv2.resize(frame, (640, 360)) 
        
        # Encode the frame to send to the server
        _, buffer = cv2.imencode(".jpg", frame) 
        frame_str = base64.b64encode(buffer).decode("utf-8")

        try:
            # Send the frame to the backend for mosaic processing
            response = requests.post("http://127.0.0.1:5001/apply_mosaic", json={"frame": frame_str})
            data = response.json()

            # Decode the processed frame
            frame_str = base64.b64decode(data["frame"])
            nparr = np.frombuffer(frame_str, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)       

        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.content}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return
                
        # Convert frame to image and display in Tkinter window
        img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    show_frame() # Start displaying the video feed

    mosaic_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), mosaic_window.destroy()))

# Function3: detect features(eyes, nose, and mouth)
def detect_features():
    # Activate the backend file
    start_flask_server("detect_features.py", "http://127.0.0.1:5002")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a new window to display the video feed
    features_window = tk.Toplevel(root)
    features_window.title("Detect Features")
    lmain = tk.Label(features_window)
    lmain.pack()

    def show_frame():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return

        # Reduce size to speed up
        frame = cv2.resize(frame, (640, 320)) 

        # Encode the frame to send to the server
        _, buffer = cv2.imencode(".jpg", frame) 
        frame_str = base64.b64encode(buffer).decode("utf-8")

        try:
            # Send the frame to the backend for feature detection
            response = requests.post("http://127.0.0.1:5002/detect_features", json={"frame": frame_str})
            data = response.json()

            # Decode the processed frame
            frame_str = base64.b64decode(data["frame"])
            nparr = np.frombuffer(frame_str, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   

        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.content}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return
            
        # Convert frame to image and display in Tkinter window
        img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    show_frame() # Start displaying the video feed

    features_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), features_window.destroy()))

# Function 4: single object tracking
def single_object_tracking():
    # Activate the backend file
    start_flask_server("single_object_tracking.py", "http://127.0.0.1:5003")

    tracking = False # track state flag

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a new window to display the video feed
    tracking_window = tk.Toplevel(root)
    tracking_window.title("Single Object Tracking")
    lmain = tk.Label(tracking_window)
    lmain.pack()

    def select_roi(event=None):
        nonlocal tracking
        ret, frame = cap.read() # Use the latest frame from show_frame to select ROI
        if not ret:
            print("Cannot receive frame")
            return
        frame = cv2.resize(frame, (640, 320)) # Reduce size to speed up
        
        # Let the user select the object to track
        roi = cv2.selectROI("Select Object", frame, showCrosshair=False, fromCenter=False)

        # Encode the frame to send to the server
        _, buffer = cv2.imencode(".jpg", frame) 
        frame_str = base64.b64encode(buffer).decode("utf-8")
        roi_list = [roi[0], roi[1], roi[2], roi[3]] # (x, y, w, h)

        try:
            # Send the frame and selected ROI to the backend to initialize tracking 
            response = requests.post("http://127.0.0.1:5003/object_tracking", 
                                    json={"frame": frame_str, "roi": roi_list, "mode": "initialize"})
            
            if response.status_code == 200:
                tracking = True
            print(response.json())

        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.content}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return

    def show_frame():
        nonlocal tracking
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return

        # Reduce size to speed up
        frame = cv2.resize(frame, (640, 320)) 

        if tracking:
            # Encode the frame to send to the server
            _, buffer = cv2.imencode(".jpg", frame) 
            frame_str = base64.b64encode(buffer).decode("utf-8")            

            try:
                # Send the frame and selected ROI to the backend to initialize tracking 
                response = requests.post("http://127.0.0.1:5003/object_tracking", 
                json={"frame": frame_str, "mode": "track"})
                data = response.json()

                # Decode the processed frame
                frame_str = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_str, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  

            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.content}")
                return
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return                 
        else:
            # If not tracking, use the current frame
            processed_frame = frame

        # Convert frame to image and display in Tkinter window
        img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    # Bind the "a" key to select an object to track(only bind once)
    tracking_window.bind("<KeyPress-a>", lambda event: select_roi())
    
    show_frame()  # Start displaying the video feed

    tracking_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), tracking_window.destroy()))

# Function 5: multiple object tracking
def multi_object_tracking():
    # Activate the backend file
    start_flask_server("multi_object_tracking.py", "http://127.0.0.1:5004")

    tracking = False # Track state flag
    selected_rois = [] # Store ROIs selected by the user

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a new window to display the video feed
    tracking_window = tk.Toplevel(root)
    tracking_window.title("Multi Object Tracking")
    lmain = tk.Label(tracking_window)
    lmain.pack()

    # Label to show instructions for selecting objects
    instruction_label = tk.Label(tracking_window, text="Press 'a' to select up to 3 objects for tracking", font=("Helvetica", 12))
    instruction_label.pack(pady=10)

    # Prompt the user to select three objects
    def select_rois(event=None):
        nonlocal tracking
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return
        frame = cv2.resize(frame, (640, 320)) # Reduce size to speed up

        # Prompt the user to select three objects
        num_selected = 0
        max_objects = 3

        while num_selected < max_objects: # Select up to 3 objects for tracking
            cv2.imshow(f"Select Object {num_selected + 1} / {max_objects}", frame)
            roi = cv2.selectROI(f"Select Object {num_selected + 1} / {max_objects}", frame, showCrosshair=False, fromCenter=False)
            if roi != (0, 0, 0, 0): # Ensure valid ROI
                selected_rois.append(list(roi))
                num_selected += 1
                print(f"Selected {num_selected} objects")
            else:
                print("Invalid selection, please try again.")

            # Close the OpenCV ROI window to avoid multiple windows
            cv2.destroyAllWindows()

        # Encode the frame to send to the server
        _, buffer = cv2.imencode(".jpg", frame) 
        frame_str = base64.b64encode(buffer).decode("utf-8")

        try:              
            # Send the frame and selected ROI to the backend to initialize tracking 
            response = requests.post("http://127.0.0.1:5004/object_tracking", 
                                    json={"frame": frame_str, "rois": selected_rois, "mode": "initialize"})
            
            if response.status_code == 200:
                tracking = True
            print(response.json())

        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.content}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return

    def show_frame():
        nonlocal tracking
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            return

        # Reduce size to speed up
        frame = cv2.resize(frame, (640, 320)) 

        if tracking:
            # Encode the frame to send to the server
            _, buffer = cv2.imencode(".jpg", frame) 
            frame_str = base64.b64encode(buffer).decode("utf-8")            

            try:
                # Send the frame and selected ROI to the backend to initialize tracking 
                response = requests.post("http://127.0.0.1:5004/object_tracking", 
                json={"frame": frame_str, "mode": "track"})
                data = response.json()

                # Decode the processed frame
                frame_str = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_str, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   

            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.content}")
                return
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return
        else:
            # If not tracking, use the current frame
            processed_frame = frame

        # Convert frame to image and display in Tkinter window
        img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    # Bind the "a" key to select objects to track(only bind once)
    tracking_window.bind("<KeyPress-a>", select_rois)

    show_frame() # Start displaying the video feed

    tracking_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), tracking_window.destroy()))

# Configuration of function buttons
buttons = [
    ("Activate face recognition", face_detection, "#ffcccc"), # pink background
    ("Activate face mosaic", face_mosaic, "#ccffcc"), # light green background
    ("Activate eyes, nose, mouth detection", detect_features, "#ccccff"), # light blue background
    ("Activate single object tracking", single_object_tracking, "#ffffcc"), # light yellow background
    ("Activate multiple object tracking", multi_object_tracking, "#ffccff"), # light purple background
]

# Put buttons
for text, command, color in buttons:
    btn = create_button(text, command, color)
    btn.pack(pady=15) # The space between each button  

# Operate main loop
root.mainloop()