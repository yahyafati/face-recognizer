import face_recognition
import cv2
import os, sys
import numpy as np
import math
from face import Face
from time import sleep
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

    @staticmethod
    def get_face_by_name(faces, name):
        for face in faces:
            if face.name and face.name.lower() == name.lower():
                return face
        return None
    

    def __init__(self):
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

    def display_annotations(self, frame, face):
        face_encoding = face.encoding
        best_match_index = face.best_match_index
        (top, right, bottom, left) = face.get_location(4)
        name = face.name

        cv2.rectangle(frame, (left, top), (right, bottom), face.color, 2)
        center = face.get_center(4)
        cv2.circle(frame, center, 2, face.color, 2)


        # creat a line from the center of the face to the center of the previous face
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

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            face.best_match_index = best_match_index
            if matches[best_match_index]:
                name = self.known_faces[best_match_index].name
                confidence = face_confidence_formatted(face_distances[best_match_index])
                previous_face = FaceRecognizer.get_face_by_name(self.previous_faces, name)
                if previous_face:
                    face.previous_locations = previous_face.previous_locations.copy()
                    face.color = previous_face.color
                face.previous_locations.append(face.location)
            else:
                face.color = (0, 0, 255)
                self.unknown_faces.append(face)
            face.confidence = confidence
            face.name = name
    
    def get_small_rgb_frame(self):
        small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.25, fy=0.25)
        return small_frame[:, :, ::-1]
    
    def get_face_by_position(self, x, y):
        for face in self.faces:
            if face.is_in_face(x, y, 4):
                return face
        return None
    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)
        
        cv2.namedWindow("Video")
        mouse_callback = lambda event, x, y, flags, param: mouse_callback_handler(self, event, x, y, flags, param)
        cv2.setMouseCallback("Video", mouse_callback)
        
        while True:
            if self.paused:
                if not self.handle_key_press(cv2.waitKey(1)):
                    break
                continue
            _, frame = video_capture.read()
            self.current_frame = frame
            rgb_small_frame = self.get_small_rgb_frame()


            if self.process_this_frame:
                self.process_frame(rgb_small_frame)
            
            self.process_this_frame = not self.process_this_frame

            for face in self.faces:
                self.display_annotations(frame, face)
            
            cv2.imshow("Video", frame)

            key = cv2.waitKey(1)
            if not self.handle_key_press(key):
                break
        video_capture.release()
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

def mouse_callback_handler(face_recognizer: FaceRecognizer, event, x, y, _flags, _param):
    if face_recognizer.taking_input:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        face_recognizer.toggle_pause()
        if not face_recognizer.paused:
            return
        face = face_recognizer.get_face_by_position(x, y)
        if face:
            face.name = "Selected"
            face.color = (0, 255, 0)
            face_recognizer.display_annotations(face_recognizer.current_frame, face)
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
            face_recognizer.display_annotations(face_recognizer.current_frame, face)
            cv2.imshow("Video", face_recognizer.current_frame)
            face_recognizer.toggle_pause()

