import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from gaze_tracking import GazeTracking

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize gaze tracking and webcam
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Retrieve the screen dimensions
screen_width, screen_height = pyautogui.size()
screen_x, screen_y = screen_width // 2, screen_height // 2

# Initialize previous head position for smoothing
smoothed_x, smoothed_y = screen_x, screen_y

# Blink detection variables
blink_start_time = None
blink_duration_threshold = 0.5  # Duration to consider a blink as long (in seconds)

# Function to draw a small green X
def draw_x(img, center, size=5, color=(0, 255, 0)):
    x, y = center
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, 2)
    cv2.line(img, (x + size, y - size), (x - size, y + size), color, 2)

# Function to calculate average position of landmarks
def calculate_avg_landmark_position(landmarks, indices, frame_w, frame_h):
    avg_x, avg_y = 0, 0
    for idx in indices:
        landmark = landmarks[idx]
        avg_x += landmark.x * frame_w
        avg_y += landmark.y * frame_h
    avg_x /= len(indices)
    avg_y /= len(indices)
    return avg_x, avg_y

# Landmarks indices for full face tracking
all_landmarks_indices = list(range(468))

while True:
    # Capture a new frame from the webcam
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    # Refresh and analyze the frame with gaze tracking
    gaze.refresh(frame)
    gaze_point = gaze.get_gaze_point()
    if gaze_point:
        gaze_x, gaze_y = gaze_point
        gaze_x = min(max(gaze_x, 0), screen_width - 1)
        gaze_y = min(max(gaze_y, 0), screen_height - 1)
    else:
        gaze_x, gaze_y = None, None

    # Convert frame to RGB and process with MediaPipe Face Mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        frame_h, frame_w, _ = frame.shape

        avg_x, avg_y = calculate_avg_landmark_position(landmarks, all_landmarks_indices, frame_w, frame_h)

        # Calculate head movement and apply smoothing
        move_x = avg_x - smoothed_x
        move_y = avg_y - smoothed_y
        smoothed_x += move_x * 0.3  # Adjust smoothing factor as needed
        smoothed_y += move_y * 0.3

        smoothed_x = max(0, min(screen_width, smoothed_x))
        smoothed_y = max(0, min(screen_height, smoothed_y))

        # Move the cursor based on head tracking
        pyautogui.moveTo(smoothed_x, smoothed_y)

        # Draw landmarks
        for idx in all_landmarks_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            draw_x(frame, (x, y), size=3, color=(0, 255, 0))

    # Detect blinking and perform clicks
    if gaze.is_blinking():
        if blink_start_time is None:
            blink_start_time = time.time()
        else:
            blink_duration = time.time() - blink_start_time
            if blink_duration > blink_duration_threshold:
                pyautogui.mouseDown(button='left')  # Long blink
            else:
                pyautogui.click(button='left')  # Short blink
    else:
        if blink_start_time is not None:
            blink_duration = time.time() - blink_start_time
            if blink_duration > blink_duration_threshold:
                pyautogui.mouseUp(button='left')  # Release mouse if long blink
            blink_start_time = None

    # Update gaze direction text and continuous cursor movement
    text = ""
    if gaze_x is not None and gaze_y is not None:
        if gaze.is_right():
            text = "Looking right"
            smoothed_x += 5  # Move cursor right
        elif gaze.is_left():
            text = "Looking left"
            smoothed_x -= 5  # Move cursor left
        elif gaze.is_up():
            text = "Looking up"
            smoothed_y -= 5  # Move cursor up
        elif gaze.is_down():
            text = "Looking down"
            smoothed_y += 5  # Move cursor down
        elif gaze.is_center():
            text = "Looking center"

        # Ensure cursor stays within screen bounds
        smoothed_x = max(0, min(screen_width, smoothed_x))
        smoothed_y = max(0, min(screen_height, smoothed_y))

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Display the annotated frame
    cv2.imshow("Demo", frame)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and destroy all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
