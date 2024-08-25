from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import pickle
import os
from flask_mysqldb import MySQL
import logging

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'ayam@123'
app.config['MYSQL_DB'] = 'face_recognition'

mysql = MySQL(app)

# Define paths and parameters
DETECTOR_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/face_detection_model"
EMBEDDING_MODEL_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/openface_nn4.small2.v1.t7"
RECOGNIZER_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/output/PyPower_recognizer.pickle"
LABEL_ENCODER_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/output/PyPower_label.pickle"
CONFIDENCE = 0.8
MATCH_THRESHOLD = 0.8

# Load face detector
protoPath = os.path.sep.join([DETECTOR_PATH, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load face embedding model
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)

# Load recognizer and label encoder
recognizer = pickle.loads(open(RECOGNIZER_PATH, "rb").read())
le = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())

# Initialize video stream
vs = None

# Initialize global variables
user_identified = False
identified_user = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def login_page():
    global user_identified, identified_user, vs
    user_identified = False
    identified_user = None
    if vs is not None:
        vs.release()
    vs = cv2.VideoCapture(0)
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cur.fetchone()
    cur.close()

    if user:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Unauthorized access!'})

@app.route('/webcam')
def webcam_page():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user_verified')
def user_verified():
    username = request.args.get('username')
    # Simulated verification process
    if username == 'peeyush':
        return jsonify({'verified': True, 'user': 'peeyush'})
    elif username == 'rajat':
        return jsonify({'verified': True, 'user': 'rajat'})
    else:
        return jsonify({'verified': False, 'message': 'User not verified'})

@app.route('/user/<username>')
def user_page(username):
    return render_template(f'{username}.html')

@app.route('/not_verified_user')
def not_verified_user():
    return render_template('not_verified_user.html')

@app.route('/reset_webcam')
def reset_webcam():
    global vs
    if vs is not None:
        vs.release()
    vs = cv2.VideoCapture(0)
    return jsonify({'success': True})


def generate_frames():
    global user_identified, identified_user, vs
    while True:
        success, frame = vs.read()
        if not success:
            break
        else:
            (h, w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]

                    if proba > CONFIDENCE:
                        if proba > MATCH_THRESHOLD:
                            name = le.classes_[j]
                        else:
                            name = "unknown"
                    else:
                        name = "unknown"

                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                    if not user_identified and name != "unknown" and proba > MATCH_THRESHOLD:
                        user_identified = True
                        identified_user = name
                        vs.release()
                        logging.debug(f"Verified user {identified_user}")
                        break

            if user_identified:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
