import os
import cv2
import time
import mediapipe as mp
import numpy as np
import uuid

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

IMAGE_PATH = 'D:\PyCharmProject\DataCollector\CollectedImages'
labels = ['A', 'B', 'C',
          'D', 'E', 'F',
          'G', 'H', 'I',
          'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R',
          'S', 'T', 'U', 'V',
          'X', 'Y']
number_imgs = 20

for label in labels:
    os.mkdir('D:\PyCharmProject\DataCollector\CollectedImages\\'+label)
    cap = cv2.VideoCapture(0)
    print('Collecting image for {}'.format(label))
    time.sleep(5)

    for imgnum in range(number_imgs):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resH = hands.process(imgRGB)
        black = np.zeros(img.shape,dtype=np.uint8)
        imagename = os.path.join(IMAGE_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        if resH.multi_hand_landmarks:
            for handLms in resH.multi_hand_landmarks:
                mpDraw.draw_landmarks(black, handLms, mpHands.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(color=(0, 255, 0), thickness= 5, circle_radius= 1))
        cv2.rectangle(img, (int(0.5 * img.shape[1]), 100), (img.shape[1], int(0.8 * img.shape[0])), (255, 0, 0), 2)
        cv2.rectangle(black, (int(0.5 * img.shape[1]), 100), (img.shape[1], int(0.8 * img.shape[0])), (255, 0, 0), 2)
        roi = black[100:int(0.8*img.shape[0]), int(0.5*img.shape[1]):img.shape[1]]
        cv2.imshow('Orginal image', img)
        cv2.imwrite(imagename, roi)
        cv2.imshow('ROI', roi)
        cv2.imshow('Black', black)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.release
