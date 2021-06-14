import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mouseEvent(event, x, y, flags, param): # on mouse click add a ball
    if event == cv2.EVENT_LBUTTONDOWN:
        pass

        
#openlocal file for video out
#success, image = cap.read()
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1],image.shape[0]))
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (image.shape[1],image.shape[0]))

colors = [(219,152,52), (34, 126, 230), (182, 89, 155), (113, 204, 46), (94, 73, 52), (15, 196, 241), (60, 76, 231)]


index_fingers =[]
prev_lines = []

# reading the video from webcam
cap = cv2.VideoCapture(0) 
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in  enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if i>= len(index_fingers):
                 index_fingers.append([])
            #position of the finger tip landmark[8]
            if (hand_landmarks.landmark[8].z-5 <hand_landmarks.landmark[0].z): 
                x = int(hand_landmarks.landmark[8].x * image.shape[1])
                y = int(hand_landmarks.landmark[8].y * image.shape[0])
                index_fingers[i].append([x,y])
            else:
                line = np.array(index_fingers[i], np.int32)
                line = line.reshape((-1, 1, 2))
                prev_lines.append(line)
                index_fingers[i]=[]

    cv2.polylines(image, prev_lines, False, colors[0],8 )
    for i, finger in enumerate(index_fingers):
        #print(i, finger,end='\n')
        if len(finger) >4: 
            finger = np.array(finger, np.int32)
            finger = finger.reshape((-1, 1, 2))
            cv2.polylines(image, [finger], False, colors[1],8 )

    cv2.imshow("simulation", image)
    cv2.setMouseCallback('simulation', mouseEvent) 
    retval= cv2.waitKey(5)

    if  retval & 0xFF == ord('c'):
        index_fingers.clear()
        prev_lines.clear()

    if retval & 0xFF == ord('q'):
        break
cap.release()



# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()