import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv
from flask import Flask, render_template, Response

app = Flask(__name__)

now = datetime.now()
current_date = now.strftime('%Y-%m-%d')

path = "photos"
imagePathList = os.listdir(path)
imgList = []
face_data_names = []

for img in imagePathList:
    imgList.append(cv2.imread(os.path.join(path,img)))
    face_data_names.append(img.split('.')[0])

face_data_encodings = [face_recognition.face_encodings(img)[0] for img in imgList]
students_names = face_data_names.copy()

face_encodings = []
face_locs = []
face_names = []
s = True

f = open(current_date+'.csv','w+',newline='')
writer = csv.writer(f)

def gen_frames():
    vid_source = cv2.VideoCapture(0)
    while True:
        success, frame = vid_source.read()
        if not success:
            break
        else:
            reduced_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
            rgb_reduced_frame = reduced_frame[:,:,::-1]
            if s:
                face_locs = face_recognition.face_locations(rgb_reduced_frame)
                face_encodings = face_recognition.face_encodings(rgb_reduced_frame,face_locs)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(face_data_encodings,face_encoding)
                    name = ''
                    face_dist = face_recognition.face_distance(face_data_encodings,face_encoding)
                    best_match_idx = np.argmin(face_dist)
                    if matches[best_match_idx]:
                        name = face_data_names[best_match_idx]
                    face_names.append(name)
                    if name in face_data_names:
                        if name in students_names:
                            students_names.remove(name)
                            print(f"{name} marked present")
                            current_time = now.strftime('%H-%M-%S')
                            writer.writerow([name, current_time])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    vid_source.release()
    cv2.destroyAllWindows()
    f.close()

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
