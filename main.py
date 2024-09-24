from camera_settings import load_camera_settings, apply_camera_settings
from calibration import load_calibration_data
from aruco_detection import detect_aruco_markers, calculate_cube_position
import cv2
import numpy as np

def main():
    # Učitavanje postavki kamere za gornju i bočnu kameru
    #upper_camera_settings = load_camera_settings('C:/Users/pirsi/PycharmProjects/pythonProject2/top_camera_settings.json')
    # = load_camera_settings('C:/Users/pirsi/PycharmProjects/pythonProject2/side_camera_settings.json')

    # Učitavanje kalibracijskih podataka za gornju i bočnu kameru iz direktorija calibration_data
    upper_camera_calibration = load_calibration_data('C:/Users/pirsi/PycharmProjects/pythonProject2/calibration_data/top_camera_calibration.json')
    side_camera_calibration = load_calibration_data('C:/Users/pirsi/PycharmProjects/pythonProject2/calibration_data/side_camera_calibration.json')

    # Pretvaranje camera_matrix i dist_coeffs u numpy nizove
    camera_matrix_upper = np.array(upper_camera_calibration['camera_matrix'])
    dist_coeffs_upper = np.array(upper_camera_calibration['dist_coeffs'])

    camera_matrix_side = np.array(side_camera_calibration['camera_matrix'])
    dist_coeffs_side = np.array(side_camera_calibration['dist_coeffs'])

    # Inicijalizacija kamera
    cap_upper = cv2.VideoCapture(1)
    cap_side = cv2.VideoCapture(2)

    # Primjena postavki na kamere
    apply_camera_settings(cap_upper, upper_camera_settings)
    apply_camera_settings(cap_side, side_camera_settings)

    while True:
        # Čitanje frameova s kamera
        ret_upper, frame_upper = cap_upper.read()
        ret_side, frame_side = cap_side.read()

        if not ret_upper or not ret_side:
            print("Neuspješno čitanje framea.")
            break

        # Detekcija ArUco markera na gornjoj i bočnoj kameri
        corners_upper, ids_upper, rvecs_upper, tvecs_upper = detect_aruco_markers(frame_upper, upper_camera_calibration)
        corners_side, ids_side, rvecs_side, tvecs_side = detect_aruco_markers(frame_side, side_camera_calibration)

        # Prikaz detektiranih markera na gornjoj kameri
        if ids_upper is not None and len(ids_upper) > 0:
            cv2.aruco.drawDetectedMarkers(frame_upper, corners_upper, ids_upper)
        if ids_side is not None and len(ids_side) > 0:
            cv2.aruco.drawDetectedMarkers(frame_side, corners_side, ids_side)

        # Prikaz koordinatnih osi na referentnom ArUco markeru (ID 120) na gornjoj kameri
        if ids_upper is not None and 120 in ids_upper:
            idx_ref_upper = np.where(ids_upper == 120)[0][0]
            cv2.drawFrameAxes(frame_upper, camera_matrix_upper, dist_coeffs_upper,
                              rvecs_upper[idx_ref_upper], tvecs_upper[idx_ref_upper], 0.1)  # Osi dužine 10 cm

        if ids_side is not None and 120 in ids_side:
            idx_ref_side = np.where(ids_side == 120)[0][0]
            cv2.drawFrameAxes(frame_side, camera_matrix_side, dist_coeffs_side,
                              rvecs_side[idx_ref_side], tvecs_side[idx_ref_side], 0.1)  # Osi dužine 10 cm

        # Provjera jesu li markeri detektirani na obje kamere
        if ids_upper is not None and len(ids_upper) > 0 and ids_side is not None and len(ids_side) > 0:
            print(f"Detektirani markeri na obje kamere: gornja IDs={ids_upper}, bočna IDs={ids_side}")
            # Računanje pozicije kocke u prostoru
            position_3d = calculate_cube_position(ids_upper, rvecs_upper, tvecs_upper, ids_side, rvecs_side, tvecs_side)
            if position_3d is not None:
                # Inverzija predznaka za Y i Z os
                position_3d[1] = -position_3d[1]  # Inverzija Y osi
                position_3d[2] = -position_3d[2]  # Inverzija Z osi

                # Pretvaranje pozicija iz metara u centimetre
                position_3d_cm = position_3d * 100
                print(f"Pozicija kocke (x, y, z) u cm: {position_3d_cm}")
            else:
                print("Pozicija kocke nije izračunata")
        else:
            print("Markeri nisu detektirani na obje kamere.")

        # Prikaz frameova s detektiranim markerima
        cv2.imshow('Upper Camera', frame_upper)
        cv2.imshow('Side Camera', frame_side)

        # Pritisni 'q' za izlaz iz petlje
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Oslobađanje resursa
    cap_upper.release()
    cap_side.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
