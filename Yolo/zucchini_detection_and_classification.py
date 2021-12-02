import numpy as np
import cv2
import time
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

height = 0
show_ratio = 1.0
title_name = 'Custom Yolo'
#video 창 원하는 크기 조절, height는 비율로 설정.
width = 1000
min_confidence = 0.5

# Load Yolo
net = cv2.dnn.readNet("backup/zucchini-train-yolo_final1.weights", "train/yolov3.cfg")
classes = []
with open("train/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

 #keras 모델 생성
 def build_model():
     model = keras.Sequential([
         Conv2D(64, kernel_size = (3, 3), padding = 'same', input_shape = (256,256,3), activation = tf.nn.relu),
         MaxPooling2D(pool_size = (2, 2)),
         Dropout(0.05),

         Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu),
         MaxPooling2D(pool_size = (2, 2)),
         Dropout(0.35000000000000003),

         Flatten(),
         Dense(352, activation = tf.nn.relu),
         Dropout(0.15000000000000002),
         Dense(4, activation = tf.nn.softmax)
     ])
     return model

 #keras 모델 weight 로드
 def load_weight():
     model = build_model()
     #model.load_weights('best_model.h5')
     return model

#Main
def detectAndDisplay(image):
    h, w = image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(image, (width, height))

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    names = []
    boxes = []
    colors = []
    model = load_weight()
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
                colors.append(color_lists[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])
            color = colors[i]
            print(i, label, x, y, w, h)
            # keras 모델에 맞게 리사이징
            image = cv2.resize(image, (256, 256))
            # Input shape 맞춰주기 위해 차원 확대
            image = image[tf.newaxis,...]
            prediction = model.predict(image)
            print(np.argmax(prediction[0]))
            # Frame에 Yolo로 인식한 Zucchini 표시
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)


    cv2.imshow(title_name, img)

# video = path, webcam = 0
vs = cv2.VideoCapture('train/data/test1.mp4')

if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()
