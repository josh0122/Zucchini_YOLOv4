import numpy as np
import cv2
from glob import glob
import pandas as pd
import numpy as np
from tqdm import trange


show_ratio = 1.0
title_name = 'Custom Yolo'
min_confidence = 0.001
width = 1024
height = 683

net = cv2.dnn.readNet("yolov4-custom-baps_last.weights", "yolov4-custom-baps.cfg")
classes = []
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

df = pd.DataFrame(columns=['front1', 'front2', 'front3', 'side1', 'side2', 'side3', 'label'])

def detectAndDisplay(image, label):
    global df
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    names = []
    boxes = []
    dic = {'front1':0, 'front2':0, 'front3':0, 'side1':0, 'side2':0, 'side3':0, 'label':'0'}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            dic[names[i]] = w
    dic['label'] = label
    df = df.append(dic, ignore_index = True)


path = glob('C:\\Users\\365mc\\Desktop\\abdomen\\abdomen\\*\\*\\*.png')
for i in trange(len(path)):
    img = cv2.imread(path[i])
    detectAndDisplay(img, path[i].split('\\')[-2])
    df.to_csv('data.csv')


