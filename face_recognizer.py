import face_recognition
import cv2
import os, sys
import numpy as np
import math
from face import Face

def face_confidence(face_distance, face_match_threshold=0.6):
    _range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (_range * 2.0)
    if face_distance > face_match_threshold:
        return linear_val
    else:
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def face_confidence_formatted(face_distance, face_match_threshold=0.6):
    return "{0:.2f}%".format(face_confidence(face_distance, face_match_threshold) * 100)

class FaceRecognizer:
    process_this_frame = True
    prev_keys = []
    faces: list[Face] = []
    known_faces: list[Face] = []
    unknown_faces: list[Face] = []
    

    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        for directory in os.listdir("faces"):
            if os.path.isdir("faces/" + directory):
                self.encode_faces_directory("faces/" + directory + "/", directory)

    def encode_faces_directory(self, directory, name):
        for image in os.listdir(directory):
            if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
                face_image = face_recognition.load_image_file(directory + image)
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_faces.append(Face(name, face_encoding))
        print("Faces encoded")
    
    def get_known_face_encodings(self):
        return [face.encoding for face in self.known_faces]

    def display_annotations(self, frame, face):
        face_encoding = face.encoding
        best_match_index = face.best_match_index
        (top, right, bottom, left) = face.get_location_scaled(4)
        name = face.name

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        center = face.get_center_scaled(4)
        cv2.circle(frame, center, 2, (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        confidence = face_confidence_formatted(face_recognition.face_distance(self.get_known_face_encodings(), face_encoding)[best_match_index])
        text = name + " " + confidence
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
    
    def handle_key_press(self, key):
        unknown_name = "".join([chr(k) for k in self.prev_keys])
        unknown_name = unknown_name.strip().capitalize()
        if key == 27: # Escape
            return False
        elif key == 13: # Enter
            self.prev_keys = []
            for face in self.unknown_faces:
                face.name = unknown_name
            self.known_faces.extend(self.unknown_faces)
            print("New faces added: ", unknown_name)
        elif ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z") or key == ord(" "):
            self.prev_keys.append(key)
            print(f"\r{unknown_name}", end="")
        elif key == 127: # Backspace
            self.prev_keys = self.prev_keys[:-1]
            print(f"\r{unknown_name}", end="")
        elif key != -1:
            print(key)
        return True
    
    def process_frame(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
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
            else:
                self.unknown_faces.append(face)
            face.confidence = confidence
            face.name = name
    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error opening video capture")
            sys.exit(1)
        
        while True:
            _, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]


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
