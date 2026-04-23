import math
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def map_to_screen(x, y, frame_width, frame_height, screen_width, screen_height, roi_fraction=0.5):
    roi_w = frame_width * roi_fraction
    roi_h = frame_height * roi_fraction
    center_x = frame_width / 2
    center_y = frame_height / 2

    left = center_x - roi_w / 2
    top = center_y - roi_h / 2
    right = center_x + roi_w / 2
    bottom = center_y + roi_h / 2

    x_clamped = clamp(x, left, right)
    y_clamped = clamp(y, top, bottom)

    normalized_x = (x_clamped - left) / (right - left)
    normalized_y = (y_clamped - top) / (bottom - top)

    screen_x = normalized_x * screen_width
    screen_y = normalized_y * screen_height

    return int(screen_x), int(screen_y)


def distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


def is_hand_open(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended = 0
    for tip_index, pip_index in zip(finger_tips, finger_pips):
        if landmarks.landmark[tip_index].y < landmarks.landmark[pip_index].y:
            extended += 1
    return extended >= 4


def main():
    pyautogui.FAILSAFE = False
    screen_width, screen_height = pyautogui.size()

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    smooth_points = deque(maxlen=5)
    last_screen_pos = None
    move_threshold = 8
    last_click_time = 0.0
    click_cooldown = 0.35
    last_screenshot_time = 0.0
    screenshot_cooldown = 2.0
    is_dragging = False
    mode_text = "Bekleniyor"

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Kamera akışı alınamıyor. Lütfen kamera bağlantısını kontrol edin.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            frame_height, frame_width = frame.shape[:2]

            # El hareketini tüm çerçevede izlemek için ROI tam görüntü boyutuna ayarlandı.
            roi_fraction = 1.0

            if results.multi_hand_landmarks:
                primary_landmarks = results.multi_hand_landmarks[0]
                index_tip = primary_landmarks.landmark[8]
                index_pip = primary_landmarks.landmark[6]
                middle_tip = primary_landmarks.landmark[12]
                middle_pip = primary_landmarks.landmark[10]

                index_pos = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
                middle_pos = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))

                is_index_up = index_tip.y < index_pip.y
                is_middle_up = middle_tip.y < middle_pip.y
                pinch_distance = distance(index_pos, middle_pos)

                if is_index_up and not is_middle_up:
                    screen_x, screen_y = map_to_screen(
                        index_pos[0], index_pos[1], frame_width, frame_height, screen_width, screen_height, roi_fraction
                    )
                    smooth_points.append((screen_x, screen_y))
                    avg_x = int(sum(p[0] for p in smooth_points) / len(smooth_points))
                    avg_y = int(sum(p[1] for p in smooth_points) / len(smooth_points))
                    new_pos = (avg_x, avg_y)
                    if last_screen_pos is None or distance(last_screen_pos, new_pos) > move_threshold:
                        pyautogui.moveTo(avg_x, avg_y, duration=0)
                        last_screen_pos = new_pos
                    mode_text = "Hareket"
                else:
                    smooth_points.clear()
                    last_screen_pos = None
                    mode_text = "Bekleniyor"

                current_time = time.time()
                if len(results.multi_hand_landmarks) == 1 and is_hand_open(primary_landmarks):
                    if current_time - last_screenshot_time > screenshot_cooldown:
                        screenshot_filename = f"screenshot_{int(current_time)}.png"
                        pyautogui.screenshot(screenshot_filename)
                        last_screenshot_time = current_time
                        mode_text = "Ekran alindi"
                elif is_index_up and is_middle_up:
                    if pinch_distance < 40:
                        if not is_dragging and current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = current_time
                            mode_text = "Tıklama"
                    elif pinch_distance < 60:
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                            mode_text = "Sürükle"
                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False
                        mode_text = "Hareket"
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    if is_index_up and not is_middle_up:
                        mode_text = "Hareket"
                    else:
                        mode_text = "Bekleniyor"

                mp_drawing.draw_landmarks(
                    frame,
                    primary_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
                )
                cv2.circle(frame, index_pos, 8, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, middle_pos, 8, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, index_pos, middle_pos, (0, 255, 255), 2)
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                mode_text = "El bulunamadı"
                smooth_points.clear()

            cv2.putText(frame, f"Mod: {mode_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Cikis icin 'q' basin.", (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("AI Sanal Fare", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as error:
        print(f"Hata olustu: {error}")
    finally:
        if is_dragging:
            pyautogui.mouseUp()
        capture.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
