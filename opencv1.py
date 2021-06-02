import pymunk
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# define the space for handling physics
space = pymunk.Space()
space.gravity = 0, -500

# define balls
balls_radius = 30
balls_body =[]


# define heads
heads_radius = 100
heads = []

# a few color for drawing balls
colors = [(219,152,52), (34, 126, 230), (182, 89, 155), (113, 204, 46), (94, 73, 52), (15, 196, 241), (60, 76, 231)]


# reading the video from webcam
cap = cv2.VideoCapture(0) 

def addball(event, x, y, flags, param): # on mouse click add a ball
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_body = pymunk.Body(100.0,1666, body_type=pymunk.Body.DYNAMIC)
        ball_body.position = (x,475-y)
        shape = pymunk.Circle(ball_body, balls_radius)
        space.add(ball_body, shape)
        balls_body.append(ball_body)
        
#openlocal file for video out
success, image = cap.read()
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1],image.shape[0]))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (image.shape[1],image.shape[0]))


with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while(cap.isOpened()):
        success, image = cap.read()
        if success:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = face_detection.process(image)
            if results.detections:
                for i, detection in enumerate(results.detections):
                    if i>= len(heads):#add heads if needed!
                        head = pymunk.Body(10,1666, body_type=pymunk.Body.KINEMATIC)
                        head_shape = pymunk.Circle(head, heads_radius)
                        space.add(head, head_shape)
                        heads.append(head)
                    pos=mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                    # converting the coordinates
                    x = int(pos.x * image.shape[1])
                    y = int(image.shape[0]-pos.y * image.shape[0])
                    # update the velocity of head
                    heads[i].velocity = 10.0*(x - heads[i].position[0]), 10.0*(y - heads[i].position[1])
                        
            # getting the position of balls from physics engine and drawing
            for i, ball in enumerate(balls_body):
                xb = int(ball.position[0])
                yb = int(image.shape[0]-ball.position[1])
                cv2.circle(image, (xb, yb), balls_radius, colors[i%len(colors)], -1)
            
            # take a simulation step
            space.step(0.02)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            
            out.write(image)

            cv2.imshow("simulation", image)
            cv2.setMouseCallback('simulation', addball) # on mouse click add a ball
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()