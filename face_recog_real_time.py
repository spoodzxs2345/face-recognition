from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import face_recognition
import pickle
import cv2
import time

#load data and haar-cascade detector
try:
    with open('C:/Users/Delsie/Desktop/projects/face_recognition/encodings', 'rb') as f:
        data = pickle.load(f)
    detector = cv2.CascadeClassifier('C:/Users/Delsie/Desktop/projects/face_recognition/haarcascade_frontalface_default.xml')
except FileNotFoundError:
    print('File not found')
    exit()

# initialize video stream
vs = VideoStream(src=0).start() # make this a comment if you are using picamera
# uncomment this comment if you are using picamera
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# count the FPS
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = 'Unknown'

        if True in matches:
            matched_indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_indexes:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    fps.update()

fps.stop()
print('Elapsed time: {:.2f}'.format(fps.elapsed()))
print('Approx. FPS: {:.2f}'.format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

