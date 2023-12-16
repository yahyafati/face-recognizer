import face_recognizer
import sys

args = sys.argv[1:]

if __name__ == "__main__":
    fr = face_recognizer.FaceRecognizer()
    play_flappy = False
    if len(args) > 0:
        play_flappy = args[0] == "--flappy"
    fr.play_flappy = play_flappy
    fr.init_video_capture()
    fr.run_recognition()

    if play_flappy:
        for face in fr.faces:
            print(f"{face.name}: {face.score}")