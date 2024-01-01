import src.face_recognizer as face_recognizer
import sys
import argparse

args = sys.argv[1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleFace: a simple face recognition and tracking program")

    parser.add_argument("--flappy", action="store_true", help="Play a simple timed bird game")
    parser.add_argument("--video", type=int, default=0, help="Video input number")

    args = parser.parse_args()

    print("Press 'ESC' to exit the program")

    fr = face_recognizer.FaceRecognizer()
    fr.context.play_flappy = args.flappy
    fr.context.video_number = args.video
    fr.context.init_video_capture()
    fr.run_recognition()

    if args.flappy:
        for face in fr.context.faces:
            print(f"{face.name}: {face.score}")