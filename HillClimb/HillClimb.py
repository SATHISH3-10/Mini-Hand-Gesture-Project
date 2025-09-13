import cv2
import mediapipe as mp
import keyboard
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Initialize camera
cap = cv2.VideoCapture(0)
last_action = None

print("Show 1 finger for GAS, 2 fingers for BRAKE. Press ESC to quit.")
time.sleep(2)

def count_fingers(hand_landmarks, handedness):
    fingers = []
    
    # Get handedness (Left or Right)
    hand_label = handedness.classification[0].label
    
    # Thumb detection (different logic for left vs right hand)
    if hand_label == "Right":  # Right hand
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down
    else:  # Left hand
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down
    
    # Other four fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id]-2].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down
            
    return sum(fingers), hand_label

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)  # Mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(img_rgb)
    fingers_up = 0
    action = "none"
    hand_label = "None"
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers with proper handedness detection
            fingers_up, hand_label = count_fingers(hand_landmarks, handedness)
    
    # Map fingers to actions
    if fingers_up == 1:
        action = "gas"
    elif fingers_up == 2:
        action = "brake"
        
    # Update keyboard if action changed
    if action != last_action:
        if action == "gas":
            keyboard.press('right')
            keyboard.release('left')
            print("GAS")
        elif action == "brake":
            keyboard.press('left')
            keyboard.release('right')
            print("BRAKE")
        else:
            keyboard.release('right')
            keyboard.release('left')
            print("Released")
            
        last_action = action
    
    # Display info
    cv2.putText(img, f"Hand: {hand_label}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Fingers: {fingers_up}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Action: {action}", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Gesture Control", img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()