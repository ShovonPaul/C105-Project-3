import cv2

# Create a variable body_classifier to assign the CascadeClassifiler file
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Open the video file
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Convert each frame into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass each frame to the classifier
        bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

        # Create a for loop for each x, y, w, h captured in bodies
        for (x, y, w, h) in bodies:
            # Draw a rectangle around a detected area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Use cv2.imshow() to display the frame
        cv2.imshow('Pedestrian Detection', frame)

        # Break the loop if 'q' is pressed
        if 0xFF & cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
