import numpy as np
import time
import cv2
import os
from playsound import playsound
from gtts import gTTS
from pydub import AudioSegment
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

LABELS = open("coco.names").read().strip().split("\n")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
font = cv2.FONT_HERSHEY_PLAIN
lane = net.getLayerNames()
lane = [lane[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize
cap = cv2.VideoCapture(0)
ground_truth = []
predictions = []
frames = []
frame_count = 0
start = time.time()
first = True
flag = 1
output_counter = 1 

while True:
    frame_count += 1
    ret, frame = cap.read()
    frames.append(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if ret:
        key = cv2.waitKey(1)
        if frame_count % 60 == 0:
            end = time.time()
            (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(lane)
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            # Loop over each of the layer outputs
            for output in layerOutputs:
                # Loop over each of the detections
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        # Scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Use the center (x, y)-coordinates to derive the top and
                        # left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # Update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            # Apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            for i in range(len(boxes)):
                if i in idxs:
                    x, y, w, h = boxes[i]
                    label = str(classes[classIDs[i]])
                    confidence = confidences[i]
                    color = colors[classIDs[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                    predictions.append([x, y, x + w, y + h])

            texts = ["You have in front of you:"]
            if len(idxs) > 0:
                for i in idxs.flatten():
                    centerX, centerY = centers[i][0], centers[i][1]

                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])
                    flag = 0

            print(texts)

            if (flag == 0):
                description = ', '.join(texts)
                output_filename = f"C:/Users/kiran/OneDrive/Desktop/Voice-Object-detection/Voice-Object-detection{output_counter}.mp3"
                tts = gTTS(description, lang='en')
                tts.save(output_filename)
                playsound(output_filename)
                output_counter += 1 
            cv2.imshow('Frame', frame)
    
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

