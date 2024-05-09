import cv2

# Specify the full path to the cascade file
cascade_path = 'E:\Face Detection\Face_Detection-using-Opencv\haarcascade_frontalface_default.xml'

# Initialize the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# Start webcam capture
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
