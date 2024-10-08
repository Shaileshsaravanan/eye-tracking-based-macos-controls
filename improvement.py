import cv2
import mediapipe as mp
import pyautogui

# Accessing camera
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Screen dimensions
screen_w, screen_h = pyautogui.size()
screen_x, screen_y = screen_w // 2, screen_h // 2

# Initial head center position
initial_head_center = None

# Sensitivity of the mouse movement
sensitivity = 2.0  # Adjust this value to change cursor sensitivity

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        frame_h, frame_w, _ = frame.shape

        # Define the indices of landmarks to use for head tracking
        head_landmarks_indices = [1, 33, 263, 61, 291]  # Nose tip, left eyebrow, right eyebrow, left cheek, right cheek

        # Calculate the average position of the selected landmarks
        avg_x = 0
        avg_y = 0
        for idx in head_landmarks_indices:
            landmark = landmarks[idx]
            avg_x += landmark.x * frame_w
            avg_y += landmark.y * frame_h

        avg_x /= len(head_landmarks_indices)
        avg_y /= len(head_landmarks_indices)

        # If initial head center is not set, set it to the current average position
        if initial_head_center is None:
            initial_head_center = (avg_x, avg_y)

        # Calculate head movement from the initial center
        move_x = avg_x - initial_head_center[0]
        move_y = avg_y - initial_head_center[1]

        # Apply sensitivity factor
        screen_x += move_x * sensitivity
        screen_y += move_y * sensitivity

        # Smooth cursor movement
        screen_x = max(0, min(screen_w, screen_x))
        screen_y = max(0, min(screen_h, screen_y))

        pyautogui.moveTo(screen_x, screen_y)

        # Draw selected landmarks and the initial center on the frame
        for idx in head_landmarks_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        if initial_head_center:
            cv2.circle(frame, (int(initial_head_center[0]), int(initial_head_center[1])), 5, (0, 0, 255), -1)
    
    cv2.imshow("Head Tracking Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
