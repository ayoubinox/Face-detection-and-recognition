import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = "ImageAttendance"
Images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
def find_encodings(Images):
    encodList = []
    for img in Images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodList.append(encode)
    return encodList

def Mark_Attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateStr = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dateStr}")



encodeListKnown = find_encodings(Images)
print("Complete Encodings")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_resized = cv2.resize(img,(0,0),None,0.5,0.5)
    img_resized = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(img_resized)
    encodedCurFrame = face_recognition.face_encodings(img_resized,facesCurFrame)

    for encodeFace, FaceLoc in zip(encodedCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            Mark_Attendance(name)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)





