import cv2
import numpy as np

def detect_aruco_markers(frame, calibration_data):
    # Učitavanje ArUco detektora
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # Detekcija ArUco markera
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    # Ako su markeri detektirani, računamo pozicije
    if ids is not None:
        camera_matrix = np.array(calibration_data['camera_matrix'])  # Pretvorba u numpy array
        dist_coeffs = np.array(calibration_data['dist_coeffs'])  # Pretvorba u numpy array

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        return corners, ids, rvecs, tvecs
    else:
        return None, None, None, None

def calculate_cube_position(ids_upper, rvecs_upper, tvecs_upper, ids_side, rvecs_side, tvecs_side):
    # Pretvaranje 2D niza u 1D za obje kamere
    ids_upper = np.ravel(ids_upper)
    ids_side = np.ravel(ids_side)

    # Provjera jesu li svi potrebni markeri prisutni (120, 130 iz gornje kamere i 125 iz bočne)
    if 120 in ids_upper and 130 in ids_upper and 125 in ids_side:
        # Pronalazak indeksa markera
        idx_ref_upper = np.where(ids_upper == 120)[0][0]
        idx_top = np.where(ids_upper == 130)[0][0]
        idx_side = np.where(ids_side == 125)[0][0]

        # Računanje relativne pozicije gornjeg markera (130) u odnosu na referentni marker (120)
        position_top = tvecs_upper[idx_top] - tvecs_upper[idx_ref_upper]

        # Transformacija pozicije bočnog markera (125) u koordinatni sustav referentnog markera
        rmat_side, _ = cv2.Rodrigues(rvecs_side[idx_side])  # Rotacijska matrica za bočnu kameru
        rmat_upper, _ = cv2.Rodrigues(rvecs_upper[idx_ref_upper])  # Rotacijska matrica za referentni marker

        # Translacija markera 125 iz bočne kamere u koordinatni sustav referentnog markera
        position_side_in_ref = rmat_upper @ (rmat_side.T @ (tvecs_side[idx_side].T - tvecs_upper[idx_ref_upper].T))

        # Kombiniranje pozicija iz obje kamere: uzimamo prosjek gornje i bočne pozicije za precizniju 3D poziciju
        position_3d = (position_top + position_side_in_ref.T) / 2.0

        # Vraćamo samo x, y, z koordinate
        return position_3d[0]  # Vraćamo x, y, z koordinate kocke
    else:
        return None
