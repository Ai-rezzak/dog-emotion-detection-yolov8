// KLASÖR İÇİNE KOYULAN RESİMLER İLE MODEL TEST ETME //

# KÜTÜPHANELER

from ultralytics import YOLO

import cv2

import cvzone

import math

import os

# YOLO modellerini yükle

model1 = YOLO("modeller/dog_detect.pt") # Köpek Tespiti için , for dog detection

model2 = YOLO("modeller/emotion_detect.pt") # Duygu tespiti için , for emotion detection

# Sınıf isimlerini güncelle

classNames = ["Happy", "Angry", "Sad","Sleeping"]

# İşlenecek ve kaydedilecek klasör yolları

input_folder = "test_data"

output_folder = "test_data/test_foto_output"

# Klasörlerin varlığını kontrol et ve oluştur

if not os.path.exists(output_folder):

os.makedirs(output_folder)

# Klasördeki her bir resmi işle

for filename in os.listdir(input_folder):

if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

img_path = os.path.join(input_folder, filename)

img = cv2.imread(img_path)


# İlk modelin sonuçlarını al

results1 = model1(img, stream=True)

for r in results1:

boxes = r.boxes

for box in boxes:

x1, y1, x2, y2 = box.xyxy[0]

x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

w, h = x2 - x1, y2 - y1

conf = math.ceil((box.conf[0] * 100))

cls = int(box.cls[0])

if cls == 16 : # "dog" sınıfını kontrol et


text = "Dog" # Eklenecek metin

fontFace = cv2.FONT_HERSHEY_SIMPLEX # Font tipi

fontScale = 1 # Font ölçeği

color = (0, 255, 0) # Metin rengi: BGR formatında (Yeşil renk)

thickness = 2 # Metin kalınlığı


#Diktörtgen çizdirme işlemi için cv2.rectangle kullanımı

cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)


#Sınıf adı yazdırma işlemi için cvzone.putTextRect kullanımı

cvzone.putTextRect(img, f'{text} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, colorT=(255, 255, 255),

colorR=(0, 0, 255), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0))


# İkinci modelin sonuçlarını al

results2 = model2(img, stream=True)

for r in results2:

boxes = r.boxes

for box in boxes:

x1, y1, x2, y2 = box.xyxy[0]

x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

conf = math.ceil((box.conf[0] * 100))

cls = int(box.cls[0])

if cls < len(classNames):


#Dog detect modeli için diktörtgen çizdirme işlemi için cv2.rectangle kullanımı

cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)


#Dog detect modeli için için cvzone.putTextRect kullanımı

cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2,

colorT=(255, 255, 255), colorR=(0, 255,0), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0))


else:

print(f"Warning: Class index {cls} out of range for classNames")

# İşlenmiş resmi kaydet

output_path = os.path.join(output_folder, filename)

cv2.imwrite(output_path, img)