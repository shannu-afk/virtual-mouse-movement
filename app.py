import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Smooth cursor movement
prev_x, prev_y = 0, 0
smoothening = 7

# Click delay handling
click_delay = 0.3
last_left_click = 0
last_right_click = 0

def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Index finger tip = 8, Thumb tip = 4, Pinky tip = 20
            index_x, index_y = lm_list[8]
            thumb_x, thumb_y = lm_list[4]
            pinky_x, pinky_y = lm_list[20]

            # Move cursor with index finger
            screen_x = int(index_x * screen_width / w)
            screen_y = int(index_y * screen_height / h)

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Show pointer
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)

            # Left click gesture: thumb touching index finger
            if get_distance(index_x, index_y, thumb_x, thumb_y) < 40:
                if time.time() - last_left_click > click_delay:
                    pyautogui.click()
                    last_left_click = time.time()
                    cv2.putText(frame, 'Left Click', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right click gesture: thumb touching pinky
            if get_distance(pinky_x, pinky_y, thumb_x, thumb_y) < 40:
                if time.time() - last_right_click > click_delay:
                    pyautogui.rightClick()
                    last_right_click = time.time()
                    cv2.putText(frame, 'Right Click', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
