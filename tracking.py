import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

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

    # Get and display the coordinates of the pupils
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Get and display the screen coordinates where the user is looking
    gaze_point = gaze.get_gaze_point()
    if gaze_point:
        cv2.putText(frame, f"Gaze Point: {gaze_point}", (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display the annotated frame
    cv2.imshow("Demo", frame)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and destroy all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
