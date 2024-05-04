import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def calculate_direction(center, width, height):
    x, y = center

    if x < width / 3:
        return "Left"
    elif x > 2 * width / 3:
        return "Right"
    elif y < height / 3:
        return "Up"
    elif y > 2 * height / 3:
        return "Down"
    else:
        return "Center"

def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]  # Finger landmark ids for the tips
    count = 0
    for id in tip_ids:
        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
            count += 1
    return count

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

    hand_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not hand_detected:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    break

        if hand_detected:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * small_frame.shape[1]), \
                         int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * small_frame.shape[0])

                cv2.circle(frame, (cx * 2, cy * 2), 7, (255, 255, 255), -1)

                height, width, _ = frame.shape

                direction = calculate_direction((cx * 2, cy * 2), width, height)
                cv2.putText(frame, direction, (cx * 2, cy * 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                num_fingers = count_fingers(hand_landmarks)

                st = ""-
                if num_fingers == 1:
                    st = "Follow"
                elif num_fingers == 2:
                    st = "Hold"
                elif num_fingers == 3:
                    st = "Halt"
                cv2.putText(frame, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 230), 2)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()