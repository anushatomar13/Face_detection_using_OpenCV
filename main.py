import cv2


cascade_classifier = cv2.CascadeClassifier(r"C:\Users\anush\PycharmProjects\pythonProject8\haarcascade_smile.xml")

# for capturing the video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting eyes in the grayscale frame
    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Drawing rectangles around the detected eyes
    for (x, y, w, h) in detections:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Displaying the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
