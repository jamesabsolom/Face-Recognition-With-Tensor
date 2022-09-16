import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Loads the cascade

for filename in os.listdir('Test Images/Base'):
    img = cv2.imread(os.path.join('Test Images/Base', filename))    # Read the inputted image
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
        cv2.imshow(filename, roi)

while True:
    if cv2.waitKey() & 0xFF == ord('q'):
        break

