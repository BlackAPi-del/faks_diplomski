import cv2
import numpy as np
import mediapipe as mp

# Funkcija za detekciju ruke na obje kamere i povratak 2D koordinata zgloba ruke
def detect_hands(hand_landmarks_upper, hand_landmarks_side, frame_upper, frame_side):
    # Dobivanje koordinata zgloba ruke (landmark 0) u pikselima za obje kamere
    wrist_upper = hand_landmarks_upper.landmark[0]
    wrist_side = hand_landmarks_side.landmark[0]

    wrist_2d_upper = np.array([int(wrist_upper.x * frame_upper.shape[1]), int(wrist_upper.y * frame_upper.shape[0])])
    wrist_2d_side = np.array([int(wrist_side.x * frame_side.shape[1]), int(wrist_side.y * frame_side.shape[0])])

    # Prikaz ruke na slici (opcionalno)
    mp.solutions.drawing_utils.draw_landmarks(frame_upper, hand_landmarks_upper, mp.solutions.hands.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(frame_side, hand_landmarks_side, mp.solutions.hands.HAND_CONNECTIONS)

    return wrist_2d_upper, wrist_2d_side

# Funkcija za triangulaciju i izračun 3D pozicije ruke
def calculate_3d_hand_position(wrist_2d_upper, wrist_2d_side, proj_matrix_upper, proj_matrix_side):
    # Pretvaranje 2D koordinata ruke u 3D prostor pomoću obje kamere (stereoskopski pogled)
    wrist_3d = cv2.triangulatePoints(proj_matrix_upper, proj_matrix_side, wrist_2d_upper, wrist_2d_side)
    wrist_3d /= wrist_3d[3]  # Homogena koordinata u 3D

    return wrist_3d[:3]  # Vraćamo 3D koordinate ruke (x, y, z)
