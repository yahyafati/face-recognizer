import cv2
import numpy as np
from .face import Face
from time import time

class Context:
    process_this_frame = True
    prev_keys = []
    faces: list[Face] = []
    known_faces: list[Face] = []
    unknown_faces: list[Face] = []

    previous_faces: list[Face] = []
    paused = False
    taking_input = False
    current_frame = None
    video_capture = None
    food_position = (400, 400)
    play_flappy = False
    video_number = 0
    start_button_location = (300, 300)
    start_pressed = False
    start_time = None
    end_time = None

    def __init__(self):
        self.overlay_duck_image = cv2.imread("assets/flappy.png", cv2.IMREAD_UNCHANGED)
        self.overlay_duck_image = cv2.resize(self.overlay_duck_image, (0, 0), fx=0.1, fy=0.1)
        self.overlay_apple_image = cv2.imread("assets/apple.png", cv2.IMREAD_UNCHANGED)
        self.overlay_apple_image = cv2.resize(self.overlay_apple_image, (0, 0), fx=0.05, fy=0.05)
        self.rng = np.random.default_rng(int(time()) % 1000000)
    
    def get_known_face_encodings(self):
        return [face.encoding for face in self.known_faces]
    
    def get_by_name(self, name):
        return FaceRecognizer.get_face_by_name(self.faces, name)
    
    def get_small_rgb_frame(self):
        small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.25, fy=0.25)
        return small_frame[:, :, ::-1]
    
    def get_face_by_position(self, x, y):
        for face in self.faces:
            if face.is_in_face(x, y, 4):
                return face
        return None
    
    def init_video_capture(self):
        self.video_capture = cv2.VideoCapture(self.video_number)
        if not self.video_capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)
    
    def next_frame(self):
        _, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        if self.play_flappy:
            darkness_factor = 1
            frame = cv2.addWeighted(frame, darkness_factor, np.zeros(frame.shape, frame.dtype), 0, 0)
        self.current_frame = frame
        return frame

    def generate_food_position(self, frame):
        frame_height, frame_width, _ = frame.shape
        padding = 350
        rint_x = self.rng.integers(low=padding, high=frame_width - padding, size=1)
        rint_y = self.rng.integers(low=padding, high=frame_height - padding, size=1)
        self.food_position = (rint_x[0], rint_y[0])