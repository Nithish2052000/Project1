from flask import Flask, render_template, Response, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import threading
import os
import cv2 as cv
import numpy as np
import smtplib
import os
import imghdr
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        self.out = cv.VideoWriter('./videos/video.avi',fourcc, 20.0, (640,480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv.VideoCapture(0)
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            ret, jpeg = cv.imencode('.jpg', frame)
            return jpeg.tobytes()
      
        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()

ENV = 'prod'
if ENV == 'dev':
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5433/flask'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://pbwimvyhgivhtp:ec760bb4283b0fa9f054df1ef41bc81b416c92c6c220e5849059019011f308c6@ec2-3-211-167-220.compute-1.amazonaws.com:5432/d20s6copnoftgu'
app.config['SQLALCHEMY_TRACK_MODTIFICATION'] = False
db = SQLAlchemy(app)
name = ""
emailID = ""
video_camera = None
global_frame = None

class User(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(200), unique = True)
    email = db.Column(db.String(200), unique = True)
    password = db.Column(db.String(200), unique = True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password


@app.route('/')
def redirect():
    return render_template('redirect.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global name
    if request.method == 'POST':
        username    = request.values.get('Name')
        email       = request.values.get('e-mail')
        password = request.values.get('password')
        name = username
        if db.session.query(User).filter(User.username == username).count() == 0 and db.session.query(User).filter(User.email == email).count() == 0 and db.session.query(User).filter(User.password == password).count() == 0:
            data = User(username, email, password)
            db.session.add(data)
            db.session.commit()
            return render_template('index.html', name=username)
        else:
            return render_template('register.html', message = "You are seeing this because you are already registered to my website or please fill all the fields with unique values")     
    else:    
        return render_template('register.html')


@app.route('/simple')
def simple():
    frameCapture()
    return render_template('simple.html')

@app.route('/blank')
def blank():
    create_train()
    return render_template('blank.html')

@app.route('/checker')
def checker():
    status = use_trained()
    if status:
        return render_template('checker.html')
    else:
        mailsender()
        return render_template('failed.html')
@app.route('/verifier')
def verifier():
    return render_template('verifier.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    global emailID
    if request.method == 'POST':
        UserName = request.values.get('username')
        Password = request.values.get('Password')
        user_object = User.query.filter_by(username = UserName).first()
        if user_object == None:
            message = "Username or password is incorrect"
            return render_template('login.html', message = message)
        elif Password != user_object.password:
            message = "Incorrect password"
            return render_template('login.html', message = message)    
        else:
            emailID = user_object.email
            return render_template('verifier.html', name = UserName)
    else:
        return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        # FrameCapture()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def frameCapture():
    global name
    DIR = os.path.abspath('videos')
    path = os.path.join(DIR, 'video.avi')
    cam = cv.VideoCapture(path) 
    DIR1 = os.path.abspath('Mini-Images')
    
    path = os.path.join(DIR1, name)

    os.mkdir(path)
    currentframe = 0
    path = os.path.join(path, name)
    while(currentframe <= 100): 
        ret,frame = cam.read()
        if ret: 
            cv.imwrite(path+str(currentframe)+'.jpg', frame)
            currentframe += 1
        else: 
            break

    cam.release() 
    cv.destroyAllWindows() 

    
    
def create_train():
    DIR = os.path.abspath('Mini-Images')
    people = []
    for x in os.listdir(DIR):
        people.append(x)
    
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    features = []
    labels = []
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 4)

            for (x, y, a, b) in faces_rect:
                faces_roi = gray[y:y+b, x:x+a]
                features.append(faces_roi)
                labels.append(label)

    
    print('Training done-------------------------')

    feature = np.array(features)
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    print(f'Length of features = {len(features)}')
    print(f'Length of labels = {len(labels)}')                

    face_recognizer.train(features, labels)

    face_recognizer.save('face_trained.yml')
    np.save('features.npy',features)
    np.save('labels.npy', labels)

def use_trained():
    global name
    DIR = os.path.abspath('videos')
    path = os.path.join(DIR, 'video.avi')
    capture = cv.VideoCapture(path) 
    d = []
    count = 0
    while count <= 15:
        isTrue, frame = capture.read()

        haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        DIRECT = os.path.abspath('Mini-Images')
    
        people = []
        for x in os.listdir(DIRECT):
            people.append(x)

        np_load_old = np.load

        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        features = np.load('features.npy')
        labels = np.load('labels.npy')

        np.load = np_load_old

        face_recognizer = cv.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_trained.yml')

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(gray, 1.2, 4)
        for (x, y, a, b) in faces_rect:
            faces_roi = gray[y:y+b, x:x+a]

            label, confidence = face_recognizer.predict(faces_roi)
            print(f'label = {people[label]} with a confidence of {confidence}')
            d.append(confidence)
            count += 1
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    counter = 0
    for c in d:
        if c>=70:
            counter += 1
    if counter >= 8:
        path = os.path.abspath('Mini-project')
        print('The user image doesnt match')
        print(path)
        i = 0
        while(capture.isOpened() and i<=1):
            ret, frame = capture.read()
            if ret == False:
                break
            cv.imwrite('Intruder.jpg',frame)
            i+=1
        return False    
    else:
        return True
    capture.release()
    
    
def mailsender():
    EMAIL_ADDRESS = "opencvmini@gmail.com"
    EMAIL_PASSWORD = "mukemaSS"
    ImgFileName = 'Intruder.jpg'
    

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        img_data = open(ImgFileName, 'rb').read()

        subject = 'Intruder Detected'
        body = 'This person logged into your account'

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = emailID

        text = MIMEText('This person has tried to login into your account')
        msg.attach(text)
        image = MIMEImage(img_data, name = os.path.basename(ImgFileName))
        msg.attach(image)

        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_ADDRESS, emailID, msg.as_string())
        smtp.quit()

if __name__ == '__main__':
    app.run()


