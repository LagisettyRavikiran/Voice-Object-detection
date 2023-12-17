import numpy as np
import time
import cv2
import os
import imutils
from playsound import playsound
import subprocess
from gtts import gTTS
from pydub import AudioSegment

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
a = "http://172.20.44.96:8080/video"
cap = cv2.VideoCapture(a)

frame_count = 0
start = time.time()
first = True
frames = []
flag = 1
output_counter = 1  # Add a counter to track output files

while True:
    frame_count += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    frames.append(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if ret:
        key = cv2.waitKey(1)
        if frame_count % 60 == 0:
            end = time.time()
            # Grab the frame dimensions and convert it to a blob
            (H, W) = frame.shape[:2]
            # Construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                          swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(lane)

            # Initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            # Loop over each of the layer outputs
            for output in layerOutputs:
                # Loop over each of the detections
                for detection in output:
                    # Extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # Filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
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
                        boxes.append([x, y, int(width), int(height)]
                                     )
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

            texts = ["You have in front of you:"]

            # Ensure at least one detection exists
            if len(idxs) > 0:
                # Loop over the indexes we are keeping
                for i in idxs.flatten():
                    # Find positions
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
                output_filename = f"D:/a/Project/CSP{output_counter}.mp3"
                tts = gTTS(description, lang='en')
                tts.save(output_filename)
                playsound(output_filename)
                output_counter += 1  # Increment the output file counter

cap.release()
cv2.destroyAllWindows()
