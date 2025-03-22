# 🎯 Voice-Assisted Object Detection using YOLOv3  

This project utilizes **YOLOv3** for real-time object detection and integrates **text-to-speech (TTS)** functionality to provide **audio feedback** about detected objects. It helps visually impaired individuals by describing objects in their surroundings.  

## 📌 Features  
- 🎥 **Real-time object detection** using YOLOv3.  
- 🔊 **Voice output** describing detected objects with **gTTS**.  
- 🎨 **Bounding box visualization** with object labels.  
- 🚀 **Playsound integration** for instant voice feedback.  

## 🛠️ Requirements  
Ensure you have the following dependencies installed:  

```bash
pip install numpy opencv-python playsound gtts pydub matplotlib scikit-learn
```
## 🔧 Setup  

### Clone the repository:  
```bash
git clone https://github.com/your-username/Voice-Object-Detection.git
cd Voice-Object-Detection
```
### Download YOLOv3 model files:  
Download the following files and place them in the project directory:  
- [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)  
- [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)  
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)  

### Run the detection script:  
```bash
python main.py
```

## 📸 Working Demo  
- 🎥 The webcam captures real-time video.  
- 🖼️ YOLOv3 detects objects and draws bounding boxes.  
- 🔊 The detected objects are announced via audio output.  
- ⏹️ Press **'q'** to exit the program.  
 

## 🤖 Technologies Used  
- 🐍 Python  
- 🎥 OpenCV  
- 🏷️ YOLOv3  
- 🗣️ gTTS (Google Text-to-Speech)  
- 🎵 Pydub  
- 📊 Matplotlib & scikit-learn  

## 📌 Future Enhancements  
✅ Improve model accuracy using YOLOv8 or MobileNet SSD.  
✅ Add custom object detection using transfer learning.  
✅ Deploy as a mobile application for accessibility.  

## 📂 Project Structure  
```
Voice-Object-Detection/
│── yolov3.cfg
│── yolov3.weights
│── coco.names
│── object_detection.py
│── README.md
│── requirements.txt
│── demo.gif
```

## 💡 Contribution  
Feel free to fork the repo, raise issues, or contribute improvements! 🚀  

## 🏆 Credits  
Developed by [LagisettyRavikiran] ✨  
