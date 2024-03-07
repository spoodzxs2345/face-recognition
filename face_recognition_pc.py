import pickle
import face_recognition
import cv2

# load encodings from pickle file
try:
    with open('C:/Users/Delsie/Desktop/projects/face_recognition/encodings', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print('File not found')
    exit()

#test an image
image = cv2.imread('C:/Users/Delsie/Desktop/projects/face_recognition/data/Charles Dela Cruz/423619399_1468004504066590_3351359486075139957_n.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model='hog')
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

# iterate through the encodings and find the encodings that match to the input
for encoding in encodings:
    matches = face_recognition.compare_faces(data['encodings'], encoding)
    name = 'Unknown'

    if True in matches:
        matched_indexes = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matched_indexes:
            name = data['names'][i]
            counts[name] = counts.get(name, 0) + 1
        
        name = max(counts, key=counts.get) # ignore the error
    
    names.append(name)

# draw the box and the name of the person
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#adjust the size of the screen and display the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600, 800)
cv2.imshow('Image', image)
cv2.waitKey(0)