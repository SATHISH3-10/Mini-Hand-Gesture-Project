import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

options = ["rock", "paper", "scissors"]

def get_hand_sign(landmarks):
    thumb_is_open = landmarks[4].x < landmarks[3].x  # Right hand; flip for left
    fingers = []
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    if not thumb_is_open and not any(fingers):
        return "rock"
    if thumb_is_open and all(fingers):
        return "paper"
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not thumb_is_open:
        return "scissors"
    return None

cap = cv2.VideoCapture(0)
result = ""
user_sign = None
comp_choice = None
round_in_progress = False
last_result_time = 0
hand_detected_time = 0
COUNTDOWN = 2  # seconds
show_countdown = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to open webcam.")
        break
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    hand_sign = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm_list = handLms.landmark
            hand_sign = get_hand_sign(lm_list)
    else:
        hand_sign = None

    # If hand detected and not already in round or countdown, start countdown
    if not round_in_progress and hand_sign in options and not show_countdown:
        hand_detected_time = time.time()
        show_countdown = True

    # Show instructions
    if not show_countdown and not round_in_progress:
        cv2.putText(img, "Show gesture and hold steady...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Countdown logic
    if show_countdown:
        elapsed = time.time() - hand_detected_time
        if elapsed < COUNTDOWN:
            cv2.putText(img, f"Ready... {int(COUNTDOWN - elapsed)+1}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
        else:
            # Time to capture!
            user_sign = hand_sign
            comp_choice = random.choice(options)
            if user_sign == comp_choice:
                result = "Draw!"
            elif (user_sign == "rock" and comp_choice == "scissors") or \
                 (user_sign == "paper" and comp_choice == "rock") or \
                 (user_sign == "scissors" and comp_choice == "paper"):
                result = "You Win!"
            else:
                result = "You Lose!"
            last_result_time = time.time()
            round_in_progress = True
            show_countdown = False

    # Show result
    if round_in_progress:
        cv2.putText(img, f"You: {user_sign} | Computer: {comp_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(img, result, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        if (hand_sign is None) and (time.time() - last_result_time > 1):
            round_in_progress = False

    cv2.imshow("Rock Paper Scissors - Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()