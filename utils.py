import math
import os
from face import Face
import cv2
import numpy as np
from time import time


def face_confidence(face_distance, face_match_threshold=0.6):
    _range = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (_range * 2.0)
    if face_distance > face_match_threshold:
        return linear_val
    else:
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def face_confidence_formatted(face_distance, face_match_threshold=0.6):
    return "{0:.2f}%".format(face_confidence(face_distance, face_match_threshold) * 100)


def overlay_image(frame, image, x, y):
    height, width, _ = image.shape

    frame[y : y + height, x : x + width] = image[:, :, :3]


def overlay_image_transparent(frame, image, x, y):
    height, width, _ = image.shape
    x -= width // 2
    y -= height // 2

    for c in range(0, 3):
        frame[y : y + height, x : x + width, c] = image[:, :, c] * (
            image[:, :, 3] / 255.0
        ) + frame[y : y + height, x : x + width, c] * (1.0 - image[:, :, 3] / 255.0)


def handle_face_clicked(face_recognizer, face: Face):
    face_recognizer.context.next_frame()
    face.name = "Selected"
    face.color = (0, 255, 0)
    face_recognizer.display_annotations(
        face_recognizer.context.current_frame, face, no_tracking=True
    )
    cv2.imshow("Video", face_recognizer.context.current_frame)
    cv2.waitKey(1)
    face_recognizer.context.taking_input = True
    name = input("Enter name: ").strip().capitalize()
    while name == "":
        name = input("Enter name: ").strip().capitalize()
    face_recognizer.context.taking_input = False
    face.name = name
    face.color = Face.random_safe_color()
    face_recognizer.context.known_faces.append(face)
    face_recognizer.display_annotations(
        face_recognizer.context.current_frame, face, no_tracking=True
    )
    cv2.imshow("Video", face_recognizer.context.current_frame)
    face_recognizer.toggle_pause()


def mouse_callback_handler(face_recognizer, event, x, y, _flags, _param):
    if face_recognizer.context.taking_input:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if face_recognizer.context.play_flappy:
            button_location = (
                face_recognizer.context.start_button_location[0],
                face_recognizer.context.start_button_location[1],
                face_recognizer.context.start_button_location[0] + 200,
                face_recognizer.context.start_button_location[1] + 100,
            )
            if is_inside_rectangle(button_location, (x, y)):
                face_recognizer.context.start_pressed = True
            if face_recognizer.context.start_pressed:
                return
        face_recognizer.toggle_pause()
        if not face_recognizer.context.paused:
            return
        face = face_recognizer.context.get_face_by_position(x, y)
        if face:
            handle_face_clicked(face_recognizer, face)


def circles_intersect(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance < r1 + r2


def is_inside_rectangle(rectangle, point):
    x, y = point
    x1, y1, x2, y2 = rectangle
    return x1 <= x <= x2 and y1 <= y <= y2
