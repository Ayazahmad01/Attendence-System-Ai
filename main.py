import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import csv
from PIL import Image

path = r'DataSet(Direction)'
images = []
classNames = []
for file in os.listdir(path):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(path, file)
    rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    images.append(rgb)  
    classNames.append(os.path.splitext(file)[0])
print("Student Image Loaded:", classNames)
def findEncodings(images_list):
    encodeList = []
    for i, rgb in enumerate(images_list):
        print("Encoding:", classNames[i], "dtype=", rgb.dtype, "shape=", rgb.shape)  
        enc = face_recognition.face_encodings(rgb)
        if len(enc) > 0:
            encodeList.append(enc[0])
        else:
            print("No face found in:", classNames[i])
    return encodeList
encodeListKnown = findEncodings(images)
print("Encoding Completed")
if len(encodeListKnown) == 0:
    print("No encodings found. Make sure dataset images contain clear faces.")
    exit()
def attendence(name):
    today = datetime.now()
    dateString = today.strftime('%Y-%m-%d')
    filename = 'attendance.csv'
    try:
        with open(filename, 'r', newline='') as f:
            existing_data = list(csv.reader(f))
    except FileNotFoundError:
        existing_data = []
    names_today = [
        row[0] for row in existing_data
        if len(row) >= 2 and row[1].startswith(dateString)
    ]
    if name in names_today:
        return
    dtString = today.strftime('%Y-%m-%d %H:%M:%S')
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, dtString])
    print(f"Attendance marked: {name} at {dtString}")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video")
        break
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    rgbSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(rgbSmall)
    encodesCurFrame = face_recognition.face_encodings(rgbSmall, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            attendence(name)
            top, right, bottom, left = faceLoc
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# conda activate facecd
# cd /d C:\Users\Ayaz_Ahmad\Desktop\FaceDect 
# python main.py
