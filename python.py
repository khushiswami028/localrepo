import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return image


def main():
    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        frame_with_faces = detect_faces(frame)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame_with_faces)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

