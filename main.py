import face_recognizer
import sys
import argparse

args = sys.argv[1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleFace: a simple face recognition and tracking program")

    parser.add_argument("--flappy", action="store_true", help="Play a simple timed bird game")
    parser.add_argument("--video", type=int, default=0, help="Video input number")

    args = parser.parse_args()

    fr = face_recognizer.FaceRecognizer()
    play_flappy = args.flappy
    video_number = args.video
    fr.play_flappy = play_flappy
    fr.video_number = video_number
    fr.init_video_capture()
    fr.run_recognition()

    if play_flappy:
        for face in fr.faces:
            print(f"{face.name}: {face.score}")