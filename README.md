# SimpleFace: A Python-Based Facial Recognition Tool

SimpleFace is a lightweight Python tool designed for simple facial recognition and tracking tasks. This tool utilizes deep learning algorithms to detect and recognize faces in images or video streams. Additionally, the tool also tracks users faec across different frames.

Using the tracking capabilities, there is a simple game which can be played. We hope you enjoy them!


# Prerequisites
* Python 3.9 or higher installed on your machine.

# Installation
1. Clone the repository
```bash
git clone git@github.com:yahyafati/face-recognizer.git
cd face-recognizer
```
2. Create and activate a python virtual environment (`venv`) (Recommended)
```bash
python3 -m venv .
source ./bin/activate      # For macOS/Linux
.\Scripts\activate       # For Windows
```
3. Install Dependencies
```bash
./bin/pip install -r requirements.txt # For macOS/Linux
.\Scripts\pip install -r requirements.txt # For Windows
```
4. Help
```bash
./bin/python main.py --help
.\Scripts\python main.py --help
```
5. Run **Simple Face**
```bash
./bin/python main.py # For macOS/Linux
.\Scripts\python main.py # For Windows
```

Notes:
* The use of a virtual environment (simpleface_venv) is encouraged to maintain project isolation.
* Ensure you have Python 3.9 or higher installed on your system.
* The requirements.txt file contains all necessary dependencies. If any issues arise during installation, manually install the dependencies listed in this file using pip.
* Replace python simpleface.py with the appropriate command to start your SimpleFace tool based on your project's structure.

# Usage

1. Adding Known Faces:
Users can add known faces to the `faces` directory located in the project's root folder. Each person's images should be stored in a separate subdirectory within the `faces` directory. The subdirectory name could be the person's name or any identifier.
```
SimpleFace/
├── faces/
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── person2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── ...
```

2. Running the Project
```
python main.py
```
This command will initiate the facial recognition system based on the known faces provided in the `faces` directory.

3. Annotation of Known Faces
As the video starts, known faces present in the `faces` directory will be recognized and annotated with their corresponding names from the subdirectory.

4. Adding Unknown Faces
* If an unknown face appears on the screen, the face will be annotated as "Unknown". 
* Enter the person's name and press Enter and it will be added to the list of known faces

5. Exiting the App
* To exit the app press `ESC` button

0. Note
* Lighting is key. Use sufficient lighting and preferably good camera.
* Manual Addition is not persisted, i.e. won't be remembered after program is closed.

# Contributors
1. Yahya Fati Haji
2. Mohammed Jawad