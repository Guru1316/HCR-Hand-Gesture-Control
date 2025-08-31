import cv2
import mediapipe as mp
import pyautogui

# Faster key sending & avoid failsafe corner aborts
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Track key states so we don't spam
right_down = False   # gas (Right Arrow)
left_down  = False   # brake (Left Arrow)

def key_down_right():
    global right_down
    if not right_down:
        pyautogui.keyDown("right")
        right_down = True

def key_up_right():
    global right_down
    if right_down:
        pyautogui.keyUp("right")
        right_down = False

def key_down_left():
    global left_down
    if not left_down:
        pyautogui.keyDown("left")
        left_down = True

def key_up_left():
    global left_down
    if left_down:
        pyautogui.keyUp("left")
        left_down = False

def release_all():
    key_up_right()
    key_up_left()

def count_open_fingers(lm):
    """
    Return number of extended fingers (ignores thumb for simplicity/reliability).
    Uses tip vs PIP y-position: tip above PIP => finger open.
    """
    tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    pips = [6, 10, 14, 18]   # their PIP joints
    open_count = 0
    for t, p in zip(tips, pips):
        if lm[t].y < lm[p].y:
            open_count += 1
    return open_count

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    gesture = "Neutral"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        open_fingers = count_open_fingers(hand.landmark)

        if open_fingers >= 3:
            # Open hand -> GAS (Right Arrow)
            key_down_right()
            key_up_left()
            gesture = "GAS (Right Arrow)"
        elif open_fingers == 0:
            # Closed fist -> BRAKE (Left Arrow)
            key_down_left()
            key_up_right()
            gesture = "BRAKE (Left Arrow)"
        else:
            # Anything else -> release both
            release_all()
            gesture = "Neutral (release)"
    else:
        # No hand -> release both
        release_all()
        gesture = "No hand (release)"

    # HUD
    cv2.putText(frame, f"Gesture: {gesture}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, "Open hand = GAS (Right) | Fist = BRAKE (Left) | ESC to quit",
                (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("HCR Gesture Control (Right=Gas, Left=Brake)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
release_all()
cap.release()
cv2.destroyAllWindows()
