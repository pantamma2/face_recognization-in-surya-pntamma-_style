import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
facetracker_model = tf.keras.models.load_model('facetracker_model.h5', custom_objects={'localization_loss': localization_loss})

def preprocess_frame(frame):
    # Resize frame to match the input shape expected by the model
    frame_resized = cv2.resize(frame, IMAGE_SIZE)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def draw_bounding_box(frame, coords, class_id):
    height, width, _ = frame.shape
    x1, y1, x2, y2 = coords
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    # Draw bounding box
    color = (0, 255, 0) if class_id > 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw class label
    label = 'Face' if class_id > 0.5 else 'Not Face'
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Capture video from a file or webcam
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam, or provide a file path

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Get predictions from the model
    class_pred, bbox_pred = facetracker_model(preprocessed_frame)

    # Draw bounding box and class label on the frame
    draw_bounding_box(frame, bbox_pred[0], class_pred[0])

    # Display the frame
    cv2.imshow('Face Tracker', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
