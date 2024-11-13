from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from datetime import datetime


app = Flask(__name__)

cap = cv2.VideoCapture(0)

mask = cv2.imread('E:/Face-Filters/mask.png', cv2.IMREAD_UNCHANGED)
dog_filter = cv2.imread('E:/Face-Filters/flower.png', cv2.IMREAD_UNCHANGED)
orange_heart_filter = cv2.imread('E:/Face-Filters/orange-heart.png', cv2.IMREAD_UNCHANGED)

def resize_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_height = int(target_width / aspect_ratio)
        new_width = target_width

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def add_alpha_channel(image):
    if image.shape[2] == 3:  
        alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
        image = np.concatenate((image, alpha_channel), axis=2)  # Add alpha channel to image
    return image

# Apply filters based on user selection
# Tracking variables for position stabilization
last_position = None
position_buffer = []

# Define a buffer size for stabilization
BUFFER_SIZE = 10  # Number of recent positions to average for smoothing
position_buffer = []  # Store recent face positions for averaging

# Apply filters with smoother position stabilization
def apply_filter(feed, filter_type=0):
    global position_buffer
    
    # Detect faces
    gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
    feed = cv2.cvtColor(feed, cv2.COLOR_BGR2BGRA)
    face_detector = cv2.CascadeClassifier('E:/Face-Filters/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use the first detected face

        # Add the new position to the buffer
        position_buffer.append((x, y, w, h))
        
        # Keep only the last BUFFER_SIZE positions in the buffer
        if len(position_buffer) > BUFFER_SIZE:
            position_buffer.pop(0)

        # Calculate the average position and size from the buffer
        avg_x = int(sum(pos[0] for pos in position_buffer) / len(position_buffer))
        avg_y = int(sum(pos[1] for pos in position_buffer) / len(position_buffer))
        avg_w = int(sum(pos[2] for pos in position_buffer) / len(position_buffer))
        avg_h = int(sum(pos[3] for pos in position_buffer) / len(position_buffer))

        # Resize the filter based on the averaged face size
        filter_height, filter_width = int(1.2 * avg_h), int(1.2 * avg_w)
        if filter_type == 0:
            temp = cv2.resize(mask, (filter_width, filter_height))
        elif filter_type == 1:
            temp = resize_aspect_ratio(dog_filter, filter_width, filter_height)
        elif filter_type == 2:
            temp = resize_aspect_ratio(orange_heart_filter, filter_width, filter_height)

        # Position the filter above the averaged face position
        y_offset = avg_y - int(0.6 * avg_h)
        x_offset = avg_x - int(0.1 * avg_w)

        # Apply the filter onto the frame at the stabilized position
        for i in range(max(0, y_offset), min(y_offset + temp.shape[0], feed.shape[0])):
            for j in range(max(0, x_offset), min(x_offset + temp.shape[1], feed.shape[1])):
                if temp[i - y_offset, j - x_offset, 3] != 0:  # Check alpha channel
                    feed[i, j] = temp[i - y_offset, j - x_offset]

    return feed


# Stream video frames to the browser
def gen(filter_type):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the selected filter to the frame
        frame_with_filter = apply_filter(frame, filter_type)
        ret, jpeg = cv2.imencode('.jpg', frame_with_filter)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

CAPTURE_DIR = 'captured_images'

@app.route('/capture_photo/<int:filter_type>')
def capture_photo(filter_type):
    ret, frame = cap.read()
    if not ret:
        return "Failed to capture image", 500

    # Apply the selected filter to the frame
    frame_with_filter = apply_filter(frame, filter_type)

    # Get the current timestamp for the image file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(CAPTURE_DIR, f"photo_{timestamp}.png")

    # Save the captured frame (with filter applied) as an image
    cv2.imwrite(image_path, frame_with_filter)

    return f"Photo saved as {image_path}"

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:filter_type>')
def video_feed(filter_type):
    return Response(gen(filter_type), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
