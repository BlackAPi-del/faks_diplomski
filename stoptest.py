import cv2
import numpy as np
import cv2.aruco as aruco
import mediapipe as mp
import math

# Učitavanje kalibracionih parametara za top kameru iz JSON formata
calibration_data = {
    "matrix": [
        [587.4174517303443, 0.0, 331.0613996799605],
        [0.0, 588.7955507659499, 229.9914960504888],
        [0.0, 0.0, 1.0]
    ],
    "distortion": [
        [0.24470970668284586, -1.7468494149921867, -0.020415928679493836, 0.004203667353671521, 3.624409090629669]
    ]
}

# ArUco postavke
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Mediapipe za detekciju ruku
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inicijalizacija kamere
cap = cv2.VideoCapture(0)


# Funkcija za izračunavanje 2D euklidske udaljenosti
def calculate_2d_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pretvaranje slike u RGB format za Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detekcija ruke
    result = hands.process(rgb_frame)

    # Detekcija ArUco markera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Crtanje detektovanih markera na slici
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Prolaz kroz sve detektovane markere
        for corner in corners:
            # Računanje centralne tačke markera (srednja tačka svih uglova)
            marker_center = np.mean(corner[0], axis=0).astype(int)

            # Prikaz marker centra
            cv2.circle(frame, tuple(marker_center), 5, (0, 255, 0), -1)

            # Ako su ruke detektovane pomoću Mediapipe-a
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Prikazivanje tačaka ruke
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Koordinate kaziprsta (landmark #8 je vrh kaziprsta)
                    index_finger_tip = hand_landmarks.landmark[8]

                    # Pretvaranje koordinate kaziprsta iz normalizovanih vrednosti u piksele
                    h, w, _ = frame.shape
                    index_finger_x = int(index_finger_tip.x * w)
                    index_finger_y = int(index_finger_tip.y * h)

                    # Prikaz vrha kaziprsta
                    cv2.circle(frame, (index_finger_x, index_finger_y), 10, (255, 0, 0), -1)

                    # Izračunavanje udaljenosti između kaziprsta i centra markera
                    distance_2d = calculate_2d_distance((index_finger_x, index_finger_y), marker_center)

                    # Prikaz udaljenosti
                    print(f"2D Udaljenost između markera i kaziprsta: {distance_2d} pikseli")

                    # Ako je udaljenost manja od praga (npr. 50 piksela), ispisuje "STOP"
                    if distance_2d < 150:
                        print("STOP")

    # Prikaz trenutnog okvira
    cv2.imshow('Aruco Marker and Index Finger Detection', frame)

    # Izlaz iz petlje ako korisnik pritisne 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Oslobađanje resursa kamere i zatvaranje prozora
cap.release()
cv2.destroyAllWindows()
