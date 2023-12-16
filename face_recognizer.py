import face_recognition
import cv2
import os, sys
import numpy as np
import math

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
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_this_frame = True
    prev_keys = []
    unknown_faces = []
    best_match_indexes = []

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
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
        print("Faces encoded")
        print(self.known_face_names)

    def display_annotations(self, frame, face_encoding, best_match_index):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            confidence = face_confidence_formatted(face_recognition.face_distance(self.known_face_encodings, face_encoding)[best_match_index])
            text = name + " " + confidence
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
    
    def handle_key_press(self, key):
        unknown_name = "".join([chr(k) for k in self.prev_keys])
        unknown_name = unknown_name.strip().capitalize()
        if key == 27: # Escape
            return False
        elif key == 13: # Enter
            self.prev_keys = []
            self.known_face_encodings.extend(self.unknown_faces)
            self.known_face_names.extend([f"{unknown_name} {i}" for i in range(len(self.unknown_faces))])
            print("New faces added: ", unknown_name)
            print(self.known_face_names)
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
        self.face_locations = face_recognition.face_locations(frame)
        self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)

        self.face_names = []
        self.unknown_faces = []
        self.best_match_indexes = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            self.best_match_indexes.append(best_match_index)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence_formatted(face_distances[best_match_index])
            else:
                self.unknown_faces.append(face_encoding)
            
            self.face_names.append(f"{name} {confidence}")
    
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

            for face_encoding, best_match_index in zip(self.face_encodings, self.best_match_indexes):
                self.display_annotations(frame, face_encoding, best_match_index)
            
            cv2.imshow("Video", frame)

            key = cv2.waitKey(1)
            if not self.handle_key_press(key):
                break
        video_capture.release()
        cv2.destroyAllWindows()
