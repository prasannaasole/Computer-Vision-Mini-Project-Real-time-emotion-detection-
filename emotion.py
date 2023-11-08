import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

# Load the pre-trained face detector model and emotion recognition model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.hdf5', compile=False)


# Define a dictionary to map emotion labels to human-readable names
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Video Capturing
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to match the input size of the emotion recognition model
        face_roi = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the face ROI for input to the emotion recognition model
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # Predict the emotion label from the face ROI
        emotion_pred = emotion_model.predict(face_roi)[0]
        emotion_label = np.argmax(emotion_pred)

        # Draw a rectangle around the face and display the predicted emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_labels[emotion_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Wait for the 'q' key to be pressed to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
