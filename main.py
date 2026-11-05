import cv2
import mediapipe as mp
import math

class VARGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def detect_var_screen_gesture(self, hand_landmarks_list):
        if len(hand_landmarks_list) < 2:
            return False, "Waiting for 2 hands..."

        h1 = hand_landmarks_list[0]
        h2 = hand_landmarks_list[1]

        h1_index, h1_thumb = h1[8], h1[4]
        h2_index, h2_thumb = h2[8], h2[4]

        index_dist = self.calculate_distance(h1_index, h2_index)
        thumb_dist = self.calculate_distance(h1_thumb, h2_thumb)

        if index_dist < 0.25 and thumb_dist < 0.25:
            return True, "VAR REVIEW CHECK DETECTED"

        return False, "Scanning for VAR Signal..."

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[INFO] Starting AI VAR Gesture Recognition System...")
        print("[INFO] Press 'q' to exit the application.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARNING] Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            status_text = "Searching for Referee..."
            is_var_gesture = False

            if results.multi_hand_landmarks:
                all_hand_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    all_hand_landmarks.append(hand_landmarks.landmark)

                is_var_gesture, status_text = self.detect_var_screen_gesture(all_hand_landmarks)

            overlay = frame.copy()
            if is_var_gesture:
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), 12)
                cv2.putText(frame, "STATUS: VAR REVIEW IN PROGRESS", (40, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 4)
                cv2.putText(frame, f"STATUS: {status_text}", (40, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.imshow('AI Football VAR Gesture Recognizer', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = VARGestureRecognizer()
    recognizer.run()