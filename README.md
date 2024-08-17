# Eye-Tracking Based MacOS Controls

This repository provides an innovative solution for controlling your MacOS system using eye tracking, head movement, gaze detection, and speech commands. The aim is to offer a more accessible way to interact with your computer, especially useful for individuals with disabilities.

## Features

- **Head Tracking**: Control the mouse cursor by moving your head. The head position is tracked in real-time and translated into cursor movement on the screen.
- **Gaze Tracking**: Move the cursor based on where you are looking. This feature allows for fine control of the cursor based on your gaze direction.
- **Blink Detection**: Detect blinks and use them to trigger mouse clicks. Short blinks are registered as regular clicks, while longer blinks are interpreted as a long click.
- **Speech Commands**: Control your computer using voice commands. This includes typing text based on recognized speech and special commands to exit the program.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shaileshsaravanan/eye-tracking-based-macos-controls.git
   cd eye-tracking-based-macos-controls
   ```

2. **Install Dependencies**:
   Make sure you have Python 3 installed. Then, install the required Python packages:
   ```bash
   pip install opencv-python mediapipe pyautogui numpy gaze_tracking SpeechRecognition
   ```

3. **Additional Setup**:
   - Ensure you have a working webcam connected to your system.
   - Ensure your microphone is functioning and accessible.

## Usage

1. **Run the Program**:
   Execute the main script to start the application:
   ```bash
   python main.py
   ```

2. **Controls**:
   - **Head Tracking**: Move your head to control the mouse cursor.
   - **Gaze Tracking**: Look around to move the cursor based on your gaze.
   - **Blink Detection**: Blink to click; longer blinks will perform a long click.
   - **Speech Commands**: Speak to type text and use voice commands to interact with your system. Saying "end" three times consecutively will terminate the program.

## Accessibility Benefits

This tool is designed to assist users with disabilities in several ways:
- **Enhanced Control**: Allows for hands-free control of the computer, beneficial for users with limited mobility.
- **Voice Commands**: Facilitates interaction through speech, which can be particularly useful for individuals who are unable to use traditional input devices.
- **Customizable Sensitivity**: Users can adjust the sensitivity and responsiveness of the tracking features to suit their needs.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision capabilities.
- [MediaPipe](https://mediapipe.dev/) for face mesh and landmark detection.
- [GazeTracking](https://github.com/antoinelame/gaze_tracking) for gaze tracking functionality.
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for speech-to-text capabilities.

---

For any questions or issues, please contact [shaileshsaravanan](https://github.com/shaileshsaravanan).
```

### Notes:
- **Replace `main.py`** with the actual script name if it's different.
- **Adjust dependency installation commands** if any additional setup or different package versions are required.
- **Include any additional setup instructions** if your project requires more configuration.
