import face_recognition
import cv2
import os, sys
import numpy as np
import math
from face import Face
from time import sleep, time
import copy

class FaceRecognizer:
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
    start_button_location = (300, 300)
    start_pressed = False

    @staticmethod
    def get_face_by_name(faces, name):
        for face in faces:
            if face.name and face.name.lower() == name.lower():
                return face
        return None
    
    @staticmethod
    def circles_intersect(circle1, circle2):
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance < r1 + r2
    
    @staticmethod
    def is_inside_rectangle(rectangle, point):
        x, y = point
        x1, y1, x2, y2 = rectangle
        return x1 <= x <= x2 and y1 <= y <= y2
    

    def __init__(self):
        self.overlay_duck_image = cv2.imread("assets/flappy.png", cv2.IMREAD_UNCHANGED)
        self.overlay_duck_image = cv2.resize(self.overlay_duck_image, (0, 0), fx=0.1, fy=0.1)
        self.overlay_apple_image = cv2.imread("assets/apple.png", cv2.IMREAD_UNCHANGED)
        self.overlay_apple_image = cv2.resize(self.overlay_apple_image, (0, 0), fx=0.05, fy=0.05)

        self.encode_faces()
    
    def encode_faces(self):
        for directory in os.listdir("faces"):
            if os.path.isdir("faces/" + directory):
                self.encode_faces_directory("faces/" + directory + "/", directory)
        print("Faces encoded")

    def encode_faces_directory(self, directory, name):
        for image in os.listdir(directory):
            if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
                face_image = face_recognition.load_image_file(directory + image)
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_faces.append(Face(name, face_encoding))
    
    def get_known_face_encodings(self):
        return [face.encoding for face in self.known_faces]

    def generate_food_position(self, frame):
        frame_height, frame_width, _ = frame.shape
        # padding of 100 px on each side
        padding = 300
        self.food_position = (np.random.randint(padding, frame_width - padding), np.random.randint(padding, frame_height - padding))

    def place_food(self, frame):
        overlay_image_transparent(frame, self.overlay_apple_image, self.food_position[0], self.food_position[1])

    def display_start_button(self, frame):
        # draw a rectangle with text "Start" at the position of the start button
        top = self.start_button_location[1]
        left = self.start_button_location[0]
        right = left + 200
        bottom = top + 100
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = "Start"
        cv2.putText(frame, text, (left + 50, bottom - 50), font, 1.0, (0, 0, 0), 2)


    def display_annotations(self, frame, face, no_tracking=False):
        if self.play_flappy and not self.start_pressed:
            self.display_start_button(frame)
            # return
        face_encoding = face.encoding
        best_match_index = face.best_match_index
        (top, right, bottom, left) = face.get_location(4)
        name = face.name

        cv2.rectangle(frame, (left, top), (right, bottom), face.color, 2)
        center = face.get_center(4)
        # cv2.circle(frame, center, 50, face.color, 2)

        if not no_tracking:
            if self.play_flappy:
                overlay_image_transparent(frame, self.overlay_duck_image, center[0] - 50, center[1] - 50)
                self.place_food(frame)
                if self.circles_intersect((center[0], center[1], 50), (self.food_position[0], self.food_position[1], 20)):
                    self.generate_food_position(frame)
                    if face.best_match_index is not None:
                        self.known_faces[face.best_match_index].score += 1
                    face.score += 1
            else:
                previous_centers = face.get_previous_centers(4)
                for i, previous_center in enumerate(previous_centers):
                    if i == 0:
                        continue
                    cv2.line(frame, previous_centers[i - 1], previous_center, face.color, 2)
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), face.color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        confidence = face_confidence_formatted(face_recognition.face_distance(self.get_known_face_encodings(), face_encoding)[best_match_index])
        text = name + " " + confidence
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

        if self.play_flappy:
            score_text = f"Score: {face.score}"
            color = (255, 255, 255)
            bold = 2
            cv2.putText(frame, score_text, (left + 6, bottom - 6 - 35), font, 1.0, color, bold)
    
    def handle_key_press(self, key):
        unknown_name = "".join([chr(k) for k in self.prev_keys])
        unknown_name = unknown_name.strip().capitalize()
        if key == 27: # Escape
            return False
        elif key != -1:
            print(key)
        return True
    
    def get_by_name(self, name):
        return FaceRecognizer.get_face_by_name(self.faces, name)
    
    def process_frame(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        self.previous_faces = self.faces.copy()
        self.faces = [Face(None, encoding, location) for encoding, location in zip(face_encodings, face_locations)]
        self.unknown_faces = []

        known_face_encodings = self.get_known_face_encodings()
        for face in self.faces:
            face_encoding = face.encoding
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"
            score = 0

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            face.best_match_index = best_match_index
            if matches[best_match_index]:
                name = self.known_faces[best_match_index].name
                score = self.known_faces[best_match_index].score
                confidence = face_confidence_formatted(face_distances[best_match_index])
                previous_face = FaceRecognizer.get_face_by_name(self.previous_faces, name)
                if previous_face:
                    face.previous_locations = previous_face.previous_locations.copy()
                    face.color = previous_face.color
                face.previous_locations.append(face.location)
                face.previous_locations = face.previous_locations[-10:]
            else:
                face.color = (0, 0, 255)
                self.unknown_faces.append(face)
            face.confidence = confidence
            face.name = name
            face.score = score
    
    def get_small_rgb_frame(self):
        small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.25, fy=0.25)
        return small_frame[:, :, ::-1]
    
    def get_face_by_position(self, x, y):
        for face in self.faces:
            if face.is_in_face(x, y, 4):
                return face
        return None
    
    def next_frame(self):
        _, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        if self.play_flappy:
            darkness_factor = 0.4
            frame = cv2.addWeighted(frame, darkness_factor, np.zeros(frame.shape, frame.dtype), 0, 0)
        self.current_frame = frame
        return frame
    
    def write_remaining_time(self, frame, remaining_time_seconds: int):
        # at the top right corner
        top = 50
        right = frame.shape[1] - 200
        formatted = "{0:.2f} seconds".format(remaining_time_seconds)
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_DUPLEX
        bold = 2
        cv2.putText(frame, formatted, (right - 100, top + 50), font, 1.0, color, bold)
    
    def init_video_capture(self):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)

    def run_recognition(self):
        cv2.namedWindow("Video")
        mouse_callback = lambda event, x, y, flags, param: mouse_callback_handler(self, event, x, y, flags, param)
        cv2.setMouseCallback("Video", mouse_callback)
        
        start_time = None
        end_time = None
        while True:
            if self.paused:
                if not self.handle_key_press(cv2.waitKey(1)):
                    break
                continue
            frame = self.next_frame()
            rgb_small_frame = self.get_small_rgb_frame()

            if self.start_pressed and start_time is None:
                start_time = time()
                end_time = start_time + 30
                self.generate_food_position(frame)

            if self.process_this_frame:
                self.process_frame(rgb_small_frame)
            
            self.process_this_frame = not self.process_this_frame
            if self.play_flappy and self.start_pressed:
                self.write_remaining_time(frame, end_time - time())
            if self.play_flappy and self.start_pressed and time() > end_time:
                break

            for face in self.faces:
                self.display_annotations(frame, face)
            
            
            cv2.imshow("Video", frame)

            key = cv2.waitKey(1)
            if not self.handle_key_press(key):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()
    
    def toggle_pause(self):
        if not self.taking_input:
            self.paused = not self.paused
        else:
            self.paused = True
            print("Cannot pause while taking input")

def face_confidence(face_distance, face_match_threshold=0.6):
    _range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (_range * 2.0)
    if face_distance > face_match_threshold:
        return linear_val
    else:
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def face_confidence_formatted(face_distance, face_match_threshold=0.6):
    return "{0:.2f}%".format(face_confidence(face_distance, face_match_threshold) * 100)

def overlay_image(frame, image, x, y):
    height, width, _ = image.shape
    
    frame[y:y + height, x:x + width] = image[:, :, :3]

def overlay_image_transparent(frame, image, x, y):
    height, width, _ = image.shape
    x -= width // 2
    y -= height // 2
    
    for c in range(0, 3):
        frame[y:y + height, x:x + width, c] = image[:, :, c] * (image[:, :, 3] / 255.0) + frame[y:y + height, x:x + width, c] * (1.0 - image[:, :, 3] / 255.0)

def mouse_callback_handler(face_recognizer: FaceRecognizer, event, x, y, _flags, _param):
    if face_recognizer.taking_input:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if face_recognizer.play_flappy:
            button_location = (face_recognizer.start_button_location[0], face_recognizer.start_button_location[1], face_recognizer.start_button_location[0] + 200, face_recognizer.start_button_location[1] + 100)
            if face_recognizer.is_inside_rectangle(button_location, (x, y)):
                face_recognizer.start_pressed = True
            if face_recognizer.start_pressed:
                return
        face_recognizer.toggle_pause()
        if not face_recognizer.paused:
            return
        face = face_recognizer.get_face_by_position(x, y)
        if face:
            face_recognizer.next_frame()
            face.name = "Selected"
            face.color = (0, 255, 0)
            face_recognizer.display_annotations(face_recognizer.current_frame, face, no_tracking=True)
            cv2.imshow("Video", face_recognizer.current_frame)
            cv2.waitKey(1)
            face_recognizer.taking_input = True
            name = input("Enter name: ").strip().capitalize()
            while name == "":
                name = input("Enter name: ").strip().capitalize()
            face_recognizer.taking_input = False
            face.name = name
            face.color = Face.random_safe_color()
            face_recognizer.known_faces.append(face)
            face_recognizer.display_annotations(face_recognizer.current_frame, face, no_tracking=True)
            cv2.imshow("Video", face_recognizer.current_frame)
            face_recognizer.toggle_pause()

