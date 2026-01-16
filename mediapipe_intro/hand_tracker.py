import cv2
import mediapipe as mp
print("I am importing from:", mp.__file__)

try:
    print("MediaPipe Version:", mp.__version__)
except AttributeError:
    print("Could not find version number.")

# the model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# the painter
mp_draw = mp.solutions.drawing_utils

# the camera
cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Error opening video file")
else:
    fps = cap.get(5)

while (cap.isOpened()):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y:
                print("Index finger is UP!")
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if ret == True:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()