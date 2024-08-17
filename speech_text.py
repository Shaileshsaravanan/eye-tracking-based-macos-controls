import speech_recognition as sr
import pyautogui
import threading
import queue

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

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
                    response = self.recognizer.recognize_google(audio)
                    print(f"Recognized text: {response}")
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

        try:
            listen_thread.join()
            process_thread.join()
        except KeyboardInterrupt:
            # Handle termination gracefully
            self.stop_event.set()
            listen_thread.join()
            process_thread.join()

if __name__ == "__main__":
    processor = SpeechProcessor()
    processor.start()
