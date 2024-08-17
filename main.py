import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from gaze_tracking import GazeTracking
import speech_recognition as sr
import threading
import queue

# Initialize camera
cam = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize Gaze Tracking
gaze = GazeTracking()

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

# Blinking detection variables
blink_start_time = None
is_blinking = False

# Speech processing class
class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.end_counter = 0  # Counter for consecutive 'end' commands
        self.previous_commands = []  # Store last few commands

    def listen_thread(self):
        with sr.Microphone() as source:
            print("Adjusting for ambient noise, please wait...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

            while not self.stop_event.is_set():
                try:
                    # Capture audio from the microphone
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    # Continue listening in case of timeout
                    pass

    def process_thread(self):
        while not self.stop_event.is_set():
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    # Recognize speech using Google Web Speech API
                    response = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized text: {response}")
                    
                    # Store the command
                    self.previous_commands.append(response)
                    if len(self.previous_commands) > 3:
                        self.previous_commands.pop(0)

                    # Check for consecutive 'end' commands
                    if self.previous_commands == ['end', 'end', 'end']:
                        print("Detected 'end' three times consecutively. Exiting...")
                        self.stop_event.set()  # Signal to stop the program
                    else:
                        pyautogui.write(response + ' ')
                except sr.UnknownValueError:
                    # Handle the case when speech is unintelligible
                    pass
                except sr.RequestError as e:
                    # Handle the case when there's a problem with the API request
                    print(f"Sorry, there was an error with the request: {e}")

    def start(self):
        # Start the listening and processing threads
        listen_thread = threading.Thread(target=self.listen_thread)
        process_thread = threading.Thread(target=self.process_thread)

        listen_thread.start()
        process_thread.start()

        return listen_thread, process_thread

# Initialize speech processor
speech_processor = SpeechProcessor()
listen_thread, process_thread = speech_processor.start()

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

while not speech_processor.stop_event.is_set():
    # Capture a new frame from the webcam
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Refresh and analyze the frame for gaze tracking
    gaze.refresh(frame)
    gaze_point = gaze.get_gaze_point()
    
    # Process face mesh
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

    # Blink detection
    if gaze.is_blinking():
        if not is_blinking:
            blink_start_time = cv2.getTickCount()
            is_blinking = True
        else:
            blink_duration = (cv2.getTickCount() - blink_start_time) / cv2.getTickFrequency()
            if blink_duration > 1:
                pyautogui.mouseDown()  # Long click
            else:
                pyautogui.click()  # Regular click
    else:
        if is_blinking:
            is_blinking = False
            # Release the mouse button if a long click was performed
            if (cv2.getTickCount() - blink_start_time) / cv2.getTickFrequency() > 1:
                pyautogui.mouseUp()

    # Annotate the frame with gaze information
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

    if gaze_point:
        cv2.putText(frame, f"Gaze Point: {gaze_point}", (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display the annotated frame
    cv2.imshow("Head Tracking and Blink Control", frame)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()

# Stop speech processor
speech_processor.stop_event.set()
listen_thread.join()
process_thread.join()
