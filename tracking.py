import cv2
from gaze_tracking import GazeTracking
import pyautogui

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Retrieve the screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize previous gaze point to avoid excessive cursor movements
prev_gaze_point = None

while True:
    # Capture a new frame from the webcam
    _, frame = webcam.read()
    
    # Refresh and analyze the frame
    gaze.refresh(frame)
    
    # Annotate the frame with gaze information
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Get the gaze point and move the cursor accordingly
    gaze_point = gaze.get_gaze_point()
    if gaze_point:
        x_coord, y_coord = gaze_point
        
        # Ensure gaze point is within the screen dimensions
        x_coord = min(max(x_coord, 0), screen_width - 1)
        y_coord = min(max(y_coord, 0), screen_height - 1)

        # Move the cursor only if the gaze point has changed significantly
        if prev_gaze_point is None or (
            abs(prev_gaze_point[0] - x_coord) > 5 or
            abs(prev_gaze_point[1] - y_coord) > 5
        ):
            pyautogui.moveTo(x_coord, y_coord)
            prev_gaze_point = (x_coord, y_coord)

        # Display the gaze point on the frame
        cv2.putText(frame, f"Gaze Point: {gaze_point}", (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display the annotated frame
    cv2.imshow("Demo", frame)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and destroy all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
