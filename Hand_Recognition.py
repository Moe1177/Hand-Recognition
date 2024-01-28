import cv2 as cv
import mediapipe as mp
import time

# Initialize Hand Modules (mediapipe)
mp_hands = mp.solutions.hands
my_hand_drawing = mp.solutions.drawing_utils

# Chooses camera (VideoCapture(0) => first camera/ default camera)
camera = cv.VideoCapture(0)
if not camera.isOpened():
    raise Exception("Unable to find camera.")

# Create a Hands instance
hands = mp_hands.Hands()

# Loops through to continue getting the live stream
while True:
    
    _, image = camera.read()
    
    # Flip the image
    # Have to switch colors because openCV color scheme is BGR not RGB  
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB) 

    # Store the results
    start_time = time.time()
    
    results = hands.process(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Prints the FPS of the image (in console)
    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    # Checks to find the landmarks(parts of the hands)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Adds the landmarks (lines) on the hands
            my_hand_drawing.draw_landmarks(
                image,
                landmarks, mp_hands.HAND_CONNECTIONS
            )
    
    # Displays image
    cv.imshow("Hand Tracker", image)

    # Click Esc to exit program
    if cv.waitKey(1) == 27:
        break
    
    # Delays execution to assure smoothness
    time.sleep(1 / 30)

# Releases the Camera 
camera.release()
cv.destroyAllWindows()