import numpy as np
import cv2
import cv2.aruco as aruco
import json

# Parametri za ArUco markere
marker_size = 50  # veličina markera u mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()


# Učitavanje kalibracijskih podataka iz JSON datoteke
def load_calibration_data(json_file):
    with open(json_file, 'r') as f:
        calib_data = json.load(f)

    # Izvuci podatke za bočnu kameru
    mtx_side = np.array(calib_data['side_camera']['matrix']).astype(np.float32)  # Konverzija u float32
    dist_side = np.array(calib_data['side_camera']['distortion']).astype(np.float32)  # Konverzija u float32

    # Izvuci podatke za gornju kameru
    mtx_top = np.array(calib_data['top_camera']['matrix']).astype(np.float32)  # Konverzija u float32
    dist_top = np.array(calib_data['top_camera']['distortion']).astype(np.float32)  # Konverzija u float32

    # Rotacijska matrica i translacija između kamera
    R = np.array(calib_data['stereo']['R']).astype(np.float32)  # Konverzija u float32
    T = np.array(calib_data['stereo']['T']).astype(np.float32)  # Konverzija u float32

    return mtx_side, dist_side, mtx_top, dist_top, R, T


# Funkcija za detekciju ArUco markera i njihovo centriranje
def detect_aruco_marker(frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i, corner in enumerate(corners):
            # Centriranje markera
            center = np.mean(corner[0], axis=0)
            marker_id = ids[i][0]
            return marker_id, center, corners[i]
    return None, None, None


# Funkcija za procjenu pozicije ArUco markera u prostoru
def estimate_marker_pose(corner, camera_matrix, dist_coeffs):
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, marker_size, camera_matrix, dist_coeffs)
    return rvec, tvec


# Funkcija za triangulaciju
def triangulate_3d_position(pt1, pt2, mtx1, mtx2, R, T):
    P1 = np.dot(mtx1, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Bočna kamera
    P2 = np.dot(mtx2, np.hstack((R, T)))  # Gornja kamera

    pts_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
    pts_3d = pts_4d[:3] / pts_4d[3]
    return pts_3d


# Glavna petlja za detekciju i triangulaciju
def detect_and_triangulate(cap_side, cap_top, mtx_side, dist_side, mtx_top, dist_top, R, T):
    while True:
        ret_side, frame_side = cap_side.read()
        if not ret_side:
            print("Bočna kamera nije dostupna!")
            break

        ret_top, frame_top = cap_top.read()
        if not ret_top:
            print("Gornja kamera nije dostupna!")
            break

        marker_id_side, center_side, corners_side = detect_aruco_marker(frame_side, mtx_side, dist_side)
        if marker_id_side == 125:
            cv2.circle(frame_side, tuple(center_side.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(frame_side, f"Marker 125 detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rvec_side, tvec_side = estimate_marker_pose(corners_side, mtx_side, dist_side)
            cv2.drawFrameAxes(frame_side, mtx_side, dist_side, rvec_side, tvec_side, 50)

        marker_id_top, center_top, corners_top = detect_aruco_marker(frame_top, mtx_top, dist_top)
        if marker_id_top == 130:
            cv2.circle(frame_top, tuple(center_top.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(frame_top, f"Marker 130 detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rvec_top, tvec_top = estimate_marker_pose(corners_top, mtx_top, dist_top)
            cv2.drawFrameAxes(frame_top, mtx_top, dist_top, rvec_top, tvec_top, 50)

        if marker_id_side == 125 and marker_id_top == 130:
            pt_side = np.array([center_side[0], center_side[1]], dtype=np.float32).reshape(2, 1)
            pt_top = np.array([center_top[0], center_top[1]], dtype=np.float32).reshape(2, 1)

            position_3d = triangulate_3d_position(pt_side, pt_top, mtx_side, mtx_top, R, T)
            print(f"3D pozicija kocke: {position_3d}")

        cv2.imshow("Bočna kamera", frame_side)
        cv2.imshow("Gornja kamera", frame_top)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_side.release()
    cap_top.release()
    cv2.destroyAllWindows()


# Inicijalizacija kamera
cap_side = cv2.VideoCapture(1)  # Bočna kamera
cap_top = cv2.VideoCapture(0)  # Gornja kamera

# Učitavanje kalibracijskih podataka iz JSON datoteke
calibration_file = "calibration_data.json"
mtx_side, dist_side, mtx_top, dist_top, R, T = load_calibration_data(calibration_file)

# Pokretanje detekcije i triangulacije
detect_and_triangulate(cap_side, cap_top, mtx_side, dist_side, mtx_top, dist_top, R, T)
