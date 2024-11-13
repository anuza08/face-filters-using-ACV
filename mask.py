import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Function to resize the filter image while maintaining its aspect ratio
def resize_aspect_ratio(image, target_width, target_height):
    # Get the aspect ratio of the filter image
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Calculate the new width and height while maintaining aspect ratio
    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_height = int(target_width / aspect_ratio)
        new_width = target_width

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Load Haar cascades for face and eye detection
eye_detector = cv2.CascadeClassifier('E:/Face-Filters/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('E:/Face-Filters/haarcascade_frontalface_default.xml')

# Load filter images (with the possibility of not having an alpha channel)
mask = cv2.imread('E:/Face-Filters/mask.png', cv2.IMREAD_UNCHANGED)
dog_filter = cv2.imread('E:/Face-Filters/flower.png', cv2.IMREAD_UNCHANGED)

# Ensure the images have 4 channels (RGBA)
def add_alpha_channel(image):
    if image.shape[2] == 3:  # If the image has 3 channels (BGR)
        # Add an alpha channel with full opacity (255)
        alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
        image = np.concatenate((image, alpha_channel), axis=2)  # Add alpha channel to image
    return image

# Ensure filters have alpha channels
mask = add_alpha_channel(mask)
dog_filter = add_alpha_channel(dog_filter)

# Start video capture
cap = cv2.VideoCapture(0)

# Variables for active filter (0 for mask, 1 for dog filter)
active_filter = 0

# Function to switch the active filter to mask filter
def set_mask_filter(event):
    global active_filter
    active_filter = 0

# Function to switch the active filter to dog filter
def set_dog_filter(event):
    global active_filter
    active_filter = 1

# Create a figure for displaying the video feed
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust the space for buttons

# Add buttons for filter switching
ax_mask = plt.axes([0.1, 0.05, 0.35, 0.075])  # Position for Mask button
ax_dog = plt.axes([0.1, 0.15, 0.15, 0.175])  # Position for Dog button

button_mask = Button(ax_mask, 'Mask Filter')
button_mask.on_clicked(set_mask_filter)

button_dog = Button(ax_dog, 'Dog Filter')
button_dog.on_clicked(set_dog_filter)

while cap.isOpened():
    _, feed = cap.read()

    if feed is None:
        break

    gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
    feed = cv2.cvtColor(feed, cv2.COLOR_BGR2BGRA)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in faces:
        # Detect eyes within the face region
        eyes = eye_detector.detectMultiScale(gray[y:y+h, x:x+w], 1.1, 2)

        if len(eyes) < 2:
            break

        # Positioning for the filter based on the eyes' locations
        X1 = eyes[0, 0] - 50
        Y1 = eyes[0, 1] - 110
        X2 = eyes[1, 0] + eyes[1, 2] + 50
        Y2 = eyes[1, 1] + eyes[1, 3] + 20

        # Ensure that the filter is resized to fit the face region
        filter_height = Y2 - Y1 + 1
        filter_width = X2 - X1 + 1
        
        # Select the active filter based on the `active_filter` variable
        if active_filter == 0:
            # Resize and apply the mask filter
            temp = cv2.resize(mask, (filter_width, filter_height))
        elif active_filter == 1:
            # Resize and apply the dog filter with aspect ratio preservation
            dog_filter_resized = resize_aspect_ratio(dog_filter, filter_width, filter_height)
            temp = dog_filter_resized

        # Ensure the filter image doesn't go beyond the feed dimensions
        temp_height, temp_width = temp.shape[:2]
        feed_height, feed_width = feed.shape[:2]

        for i in range(max(y + Y1, 0), min(y + Y1 + temp_height, feed_height)):
            for j in range(max(x + X1, 0), min(x + X1 + temp_width, feed_width)):
                if temp[i - y - Y1, j - x - X1, 3] != 0:  # Check alpha channel
                    feed[i, j] = temp[i - y - Y1, j - x - X1]

    # Use matplotlib to display the frame
    ax.imshow(cv2.cvtColor(feed, cv2.COLOR_BGRA2RGBA))  # Convert to RGB for display
    ax.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.01)  # Short pause to allow update

    # Check for window closure
    if not plt.fignum_exists(fig.number):  # Check if window is closed
        break

# Release resources and close the plot
cap.release()
plt.close()
