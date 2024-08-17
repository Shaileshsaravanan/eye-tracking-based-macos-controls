import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Accessing camera
cam = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# Screen dimensions
screen_w, screen_h = pyautogui.size()
screen_x, screen_y = screen_w // 2, screen_h // 2

# Initial head center position
initial_head_center = None

# Sensitivity of the mouse movement
sensitivity = 2.0  # Adjust this value to change cursor sensitivity

# Smoothing parameters
alpha = 0.1  # Smoothing factor (0 < alpha < 1)
smoothed_x, smoothed_y = screen_x, screen_y

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
all_landmarks_indices = list(range(468))  # Face Mesh provides 468 landmarks

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        frame_h, frame_w, _ = frame.shape

        # Calculate the average position of the selected landmarks
        avg_x, avg_y = calculate_avg_landmark_position(landmarks, all_landmarks_indices, frame_w, frame_h)

        # If initial head center is not set, set it to the current average position
        if initial_head_center is None:
            initial_head_center = (avg_x, avg_y)

        # Calculate head movement from the initial center
        move_x = avg_x - initial_head_center[0]
        move_y = avg_y - initial_head_center[1]

        # Apply sensitivity factor
        screen_x += move_x * sensitivity
        screen_y += move_y * sensitivity

        # Apply smoothing
        smoothed_x = alpha * screen_x + (1 - alpha) * smoothed_x
        smoothed_y = alpha * screen_y + (1 - alpha) * smoothed_y

        # Clipping cursor movement
        smoothed_x = max(0, min(screen_w, smoothed_x))
        smoothed_y = max(0, min(screen_h, smoothed_y))

        pyautogui.moveTo(smoothed_x, smoothed_y)

        # Draw landmarks and the initial center on the frame
        for idx in all_landmarks_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            draw_x(frame, (x, y), size=3, color=(0, 255, 0))  # Draw small green X

        if initial_head_center:
            cv2.circle(frame, (int(initial_head_center[0]), int(initial_head_center[1])), 5, (0, 0, 255), -1)
    
    cv2.imshow("Head Tracking Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
