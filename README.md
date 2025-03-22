Voice-Assisted Object Detection using YOLOv3

📌 Overview

This project implements a real-time object detection system using YOLOv3 and provides voice feedback describing detected objects' locations in the frame. The system captures live video using a webcam, processes frames through YOLOv3, and generates audio descriptions using Google Text-to-Speech (gTTS).

🎯 Features

Real-Time Object Detection: Uses YOLOv3 for accurate and fast object detection.

Voice Feedback: Converts detected objects' descriptions into speech.

Dynamic Location Identification: Specifies whether objects are on the left, center, or right, as well as top, middle, or bottom.

Automatic Audio Generation: Saves and plays descriptions as MP3 files.

Optimized Processing: Uses Non-Maximum Suppression (NMS) to filter overlapping detections.

🚀 Installation & Setup

Prerequisites

Ensure you have Python installed (preferably Python 3.x). Install the following dependencies:

pip install numpy opencv-python playsound gtts pydub matplotlib

Clone the Repository

git clone https://github.com/your-repo/voice-object-detection.git
cd voice-object-detection

Download YOLOv3 Weights

Download the pre-trained YOLOv3 model weights:

wget https://pjreddie.com/media/files/yolov3.weights

Ensure the following files are in the project directory:

yolov3.cfg

yolov3.weights

coco.names

🛠 Usage

Run the following command to start object detection with voice assistance:

python detect.py

How It Works

Captures Frames: Captures live frames from the webcam every 60 frames.

Processes Image: Uses YOLOv3 to detect objects and their bounding boxes.

Determines Location: Identifies objects' positions within the frame.

Generates Voice Output: Converts object details into speech and plays it.

Keyboard Shortcuts

Press 'q' to exit the application.

📌 Example Output

When a person and a chair are detected in different positions, the voice output might be:

You have in front of you: bottom left person, top right chair.

🔧 Customization

Modify coco.names to detect specific objects.

Adjust confidence threshold (confidence > 0.5) in the script for different accuracy levels.

Change frame capture rate to process frames at different intervals.

🛠 Troubleshooting

Ensure yolov3.weights, yolov3.cfg, and coco.names are in the working directory.

Install missing dependencies using pip install -r requirements.txt.

📜 License

This project is open-source and available under the MIT License.

🚀 Future Improvements

✅ Integration with ESP32-CAM for portable object detection.✅ Enhanced multi-language voice descriptions.✅ Integration with Edge AI for mobile applications.

💡 Contributions & Feedback Welcome!

🔗 Stay Connected: If you have suggestions or want to contribute, feel free to open an issue or submit a pull request!
