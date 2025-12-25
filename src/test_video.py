// KLASÖR İÇİNE KOYULAN VİDEO İLE MODEL TEST ETME //

# KÜTÜPHANELERİ YÜKLEME

from ultralytics import YOLO

import cv2

import cvzone

import math

import os

# YOLO modellerini yükle

model1 = YOLO("modeller/dog_detect.pt")

model2 = YOLO("modeller/emotion.pt")

# Sınıf isimlerini güncelle

classNames = ["Happy", "Angry", "Sad","Sleeping"]

# İşlenecek ve kaydedilecek video dosyalarının yolları

input_video_path = "test_video/a.mp4"

output_video_path = "test_video//test_video_output"

# Video yakalama (capture) işlemi

cap = cv2.VideoCapture(input_video_path)

# Video yazar (writer) ayarları

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Video döngüsü

while cap.isOpened():

ret, frame = cap.read()

if not ret:

break

# İlk modelin sonuçlarını al

results1 = model1(frame, stream=True)

for r in results1:

boxes = r.boxes

for box in boxes:

x1, y1, x2, y2 = box.xyxy[0]

x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

conf = math.ceil((box.conf[0] * 100))

cls = int(box.cls[0])

name = "Dog"

if cls == 16:

cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

cvzone.putTextRect(frame, f'{name} {conf}',(max(0, x1), max(35, y1)), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0))

# İkinci modelin sonuçlarını al (sadece bu köpek için)

results2 = model2(frame[y1:y2, x1:x2], stream=True)

for r2 in results2:

boxes2 = r2.boxes

for box2 in boxes2:

x1_2, y1_2, x2_2, y2_2 = box2.xyxy[0]

x1_2, y1_2, x2_2, y2_2 = int(x1_2), int(y1_2), int(x2_2), int(y2_2)

conf2 = math.ceil((box2.conf[0] * 100))

cls2 = int(box2.cls[0])

org2 = (x1 + x1_2 + 5, y1 + y1_2 + 5)

if cls2 < len(classNames):

cv2.rectangle(frame, (x1 + x1_2, y1 + y1_2), (x1 + x2_2, y1 + y2_2), color=(0, 0, 255), thickness=2)

cvzone.putTextRect(frame, f'{classNames[cls2]} {conf2}',org2, scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0))

else:

print(f"Warning: Class index {cls2} out of range for classNames")

# Sonuçları diske kaydetme

out.write(frame)

# Sonuçları ekrana gösterme (isteğe bağlı)

cv2.imshow('frame', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):

break

# Video yakalamayı ve yazmayı serbest bırakma

cap.release()

out.release()

cv2.destroyAllWindows()