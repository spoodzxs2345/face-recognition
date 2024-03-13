import face_recognition
import pickle
import cv2
import os
from imutils import paths

known_path = list(paths.list_images('C:/Users/Delsie/Desktop/projects/face_recognition_v1/data'))

known_encodings = []
known_names = []

for (i, img_path) in enumerate(known_path):
    print('Processing image {}/{}'.format(i + 1, len(known_path)))
    name = img_path.split(os.path.sep)[-2]

    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

print('known names: ', list(known_names))

data = {'encodings': known_encodings, 'names': known_names}
f = open('C:/Users/Delsie/Desktop/projects/face_recognition_v1/encodings', 'wb')
f.write(pickle.dumps(data))
f.close()
