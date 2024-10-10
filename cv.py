import cv2

# Load pre-trained Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Optional: You can use a custom or pre-trained model for glasses detection
# glasses_cascade = cv2.CascadeClassifier('path_to_glasses_cascade.xml')

# Start video capture or use an image
cap = cv2.VideoCapture(0)  # For webcam, or use an image with cv2.imread()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Detect smile
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        
        # Optional: Detect glasses
        # glasses = glasses_cascade.detectMultiScale(roi_gray)
        # for (gx, gy, gw, gh) in glasses:
        #     cv2.rectangle(roi_color, (gx, gy), (gx+gw, gy+gh), (255, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


