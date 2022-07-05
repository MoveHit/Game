import cv2
from deepface import DeepFace

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  # processing it for our project
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
    print("Error loading xml file")

cam = cv2.VideoCapture(0)


def emotion():

    while cam.isOpened():
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 0)
            try:
                analyze = DeepFace.analyze(frame, prog_bar=False)
                user_emotion = analyze['dominant_emotion']
                # print(user_emotion)
                return user_emotion
            except:
                print("No face detected")
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            cam.release()


def main():
    user_emotion = emotion()
    print(user_emotion)


if __name__ == "__main__":
    main()
