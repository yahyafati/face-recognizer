import face_recognition
import cv2
import os, sys
import numpy as np
import math
from face import Face
from context import Context
from time import sleep, time
import copy
import utils

class FaceRecognizer:
    
    @staticmethod
    def get_face_by_name(faces, name):
        for face in faces:
            if face.name and face.name.lower() == name.lower():
                return face
        return None
    
    def __init__(self):
        self.context = Context()
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
                self.context.known_faces.append(Face(name, face_encoding))

    def place_food(self, frame):
        utils.overlay_image_transparent(frame, self.context.overlay_apple_image, self.context.food_position[0], self.context.food_position[1])

    def display_start_button(self, frame):
        # draw a rectangle with text "Start" at the position of the start button
        top = self.context.start_button_location[1]
        left = self.context.start_button_location[0]
        right = left + 200
        bottom = top + 100
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = "Start"
        cv2.putText(frame, text, (left + 50, bottom - 50), font, 1.0, (0, 0, 0), 2)


    def display_annotations(self, frame, face, no_tracking=False):
        if self.context.play_flappy and not self.context.start_pressed:
            self.display_start_button(frame)
            # return
        face_encoding = face.encoding
        best_match_index = face.best_match_index
        (top, right, bottom, left) = face.get_location(4)
        name = face.name

        cv2.rectangle(frame, (left, top), (right, bottom), face.color, 2)
        center = face.get_center(4)

        if not no_tracking:
            if self.context.play_flappy and self.context.start_pressed:
                utils.overlay_image_transparent(frame, self.context.overlay_duck_image, center[0] - 50, center[1] - 50)
                self.place_food(frame)
                if utils.circles_intersect((center[0], center[1], 50), (self.context.food_position[0], self.context.food_position[1], 20)):
                    self.context.generate_food_position(frame)
                    if face.best_match_index is not None:
                        self.context.known_faces[face.best_match_index].score += 1
                    face.score += 1
            else:
                previous_centers = face.get_previous_centers(4)
                for i, previous_center in enumerate(previous_centers):
                    if i == 0:
                        continue
                    cv2.line(frame, previous_centers[i - 1], previous_center, face.color, 2)
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), face.color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        confidence = utils.face_confidence_formatted(face_recognition.face_distance(self.context.get_known_face_encodings(), face_encoding)[best_match_index])
        text = name + " " + confidence
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

        if self.context.play_flappy:
            score_text = f"Score: {face.score}"
            color = (255, 255, 255)
            bold = 2
            cv2.putText(frame, score_text, (left + 6, bottom - 6 - 35), font, 1.0, color, bold)
    
    def handle_key_press(self, key):
        unknown_name = "".join([chr(k) for k in self.context.prev_keys])
        unknown_name = unknown_name.strip().capitalize()
        if key == 27: # Escape
            return False
        elif key != -1:
            print(key)
        return True
    
    def process_frame(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        self.context.previous_faces = self.context.faces.copy()
        self.context.faces = [Face(None, encoding, location) for encoding, location in zip(face_encodings, face_locations)]
        self.context.unknown_faces = []

        known_face_encodings = self.context.get_known_face_encodings()
        for face in self.context.faces:
            face_encoding = face.encoding
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"
            score = 0

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            face.best_match_index = best_match_index
            if matches[best_match_index]:
                name = self.context.known_faces[best_match_index].name
                score = self.context.known_faces[best_match_index].score
                confidence = utils.face_confidence_formatted(face_distances[best_match_index])
                previous_face = FaceRecognizer.get_face_by_name(self.context.previous_faces, name)
                if previous_face:
                    face.previous_locations = previous_face.previous_locations.copy()
                    face.color = previous_face.color
                face.previous_locations.append(face.location)
                face.previous_locations = face.previous_locations[-10:]
            else:
                face.color = (0, 0, 255)
                self.context.unknown_faces.append(face)
            face.confidence = confidence
            face.name = name
            face.score = score
    
    def write_remaining_time(self, frame, remaining_time_seconds: int):
        # at the top right corner
        top = 50
        right = frame.shape[1] - 200
        formatted = "{0:.2f} seconds".format(remaining_time_seconds)
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_DUPLEX
        bold = 2
        cv2.putText(frame, formatted, (right - 100, top + 50), font, 1.0, color, bold)
    
    def process_game_logic(self, frame):
        if self.context.start_pressed and self.context.start_time is None:
            self.context.start_time = time()
            self.context.end_time = self.context.start_time + 30
            self.context.generate_food_position(frame)
        if self.context.play_flappy and self.context.start_pressed:
            self.write_remaining_time(frame, self.context.end_time - time())
            if time() > self.context.end_time:
                return False
        return True

    def run_recognition(self):
        cv2.namedWindow("Video")
        mouse_callback = lambda event, x, y, flags, param: utils.mouse_callback_handler(self, event, x, y, flags, param)
        cv2.setMouseCallback("Video", mouse_callback)
        
        while True:
            if self.context.paused:
                if not self.handle_key_press(cv2.waitKey(1)):
                    break
                continue
            frame = self.context.next_frame()
            rgb_small_frame = self.context.get_small_rgb_frame()

            if self.context.process_this_frame:
                self.process_frame(rgb_small_frame)
            
            self.context.process_this_frame = not self.context.process_this_frame
            if not self.process_game_logic(frame):
                break

            for face in self.context.faces:
                self.display_annotations(frame, face)
            
            
            cv2.imshow("Video", frame)

            key = cv2.waitKey(1)
            if not self.handle_key_press(key):
                break
        self.context.video_capture.release()
        cv2.destroyAllWindows()
    
    def toggle_pause(self):
        if not self.context.taking_input:
            self.context.paused = not self.context.paused
        else:
            self.context.paused = True
            print("Cannot pause while taking input")
