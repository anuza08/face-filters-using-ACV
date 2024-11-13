import cv2
import matplotlib.pyplot as plt

# Update the paths to the full paths of the Haar cascade XML files
eye_detector = cv2.CascadeClassifier('E:/Face-Filters/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('E:/Face-Filters/haarcascade_frontalface_default.xml')

# Load the mask image
mask = cv2.imread('E:/Face-Filters/mask.png', cv2.IMREAD_UNCHANGED)

# Start video capture
cap = cv2.VideoCapture(0)

# Create a figure for displaying the video feed
fig, ax = plt.subplots()

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
        
        # Positioning for the mask based on the eyes' locations
        X1 = eyes[0, 0] - 50
        Y1 = eyes[0, 1] - 110
        X2 = eyes[1, 0] + eyes[1, 2] + 50
        Y2 = eyes[1, 1] + eyes[1, 3] + 20

        # Resize the mask to fit the face
        temp = cv2.resize(mask, (X2 - X1 + 1, Y2 - Y1 + 1))
        
        for i in range(y + Y1, y + Y2 + 1):
            for j in range(x + X1, x + X2 + 1):
                if temp[i - y - Y1, j - x - X1, 3] != 0:  # Check alpha channel
                    feed[i, j] = temp[i - y - Y1, j - x - X1]

    # Use matplotlib to display the frame
    ax.imshow(cv2.cvtColor(feed, cv2.COLOR_BGRA2RGBA))  # Convert to RGB for display
    ax.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.01)  # Short pause to allow update

    # Check for closing the window (Matplotlib will handle this)
    if not plt.fignum_exists(fig.number):  # Check if window is closed
        break

# Release resources and close the plot
cap.release()
plt.close()
