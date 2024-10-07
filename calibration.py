import numpy as np
import cv2
import json

# Parametri chessboarda (10x7 unutarnjih kutova, 11x8 kvadrata)
chessboard_size = (10, 7)
square_size = 25  # veličina kvadrata u mm

# Pripremi 3D točke (z = 0, jer su sve na ravnini)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Funkcija za prikupljanje uzoraka s jedne kamere
def collect_samples(camera_id, num_samples=10, camera_name="camera"):
    objpoints = []  # 3D točke u stvarnom svijetu
    imgpoints = []

    cap = cv2.VideoCapture(camera_id)
    collected_samples = 0

    while collected_samples < num_samples:
        ret, frame = cap.read()
        if not ret:
            print(f"{camera_name} nije dostupna!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # Prikaz vizualne potvrde
        if ret:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow(f'{camera_name}', frame)

            # Pritisni 's' za spremanje uzorka
            if cv2.waitKey(1) & 0xFF == ord('s'):
                imgpoints.append(corners)
                objpoints.append(objp)
                collected_samples += 1
                print(f"Uzeti uzorak {collected_samples} za {camera_name}")
        else:
            cv2.imshow(f'{camera_name}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return objpoints, imgpoints

# Funkcija za kalibraciju kamere i izračun greške
def calibrate_camera(objpoints, imgpoints, image_shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    # Izračunaj ukupnu reprojekcijsku grešku
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Reprojekcijska greška: {mean_error}")

    return mtx, dist, rvecs, tvecs, mean_error

# Funkcija za stereo kalibraciju i izračun stereo greške s pre-estimated parametrima
def stereo_calibration_pre_estimated(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, image_shape):
    """
    Stereo kalibracija s unaprijed definiranim unutarnjim parametrima i pretpostavljenom rotacijom i translacijom.
    """
    # Pretpostavljena rotacijska matrica - 90 stupnjeva u smjeru Z-osi
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]]).astype(np.float32)

    # Pretpostavljena translacija - gornja kamera je 500 mm iznad i 600 mm ispred bočne kamere
    T = np.array([[0], [500], [600]]).astype(np.float32)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2,
        image_shape, criteria=criteria, flags=flags, R=R, T=T
    )

    if ret:
        print("Stereo kalibracija uspješna!")

    # Izračun stereo greške
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2_proj, _ = cv2.projectPoints(objpoints[i], R, T, mtx1, dist1)
        error = cv2.norm(imgpoints1[i], imgpoints2_proj, cv2.NORM_L2) / len(imgpoints2_proj)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Stereo reprojekcijska greška: {mean_error}")

    return R, T, E, F, mean_error

# 1. Prikupljanje uzoraka i kalibracija za bočnu kameru (side camera)
print("Prikupljanje uzoraka za bočnu kameru (side). Pritisnite 's' za spremanje uzorka.")
objpoints_side, imgpoints_side = collect_samples(camera_id=1, num_samples=10,
                                                 camera_name="Bočna kamera (side)")  # Bočna kamera
image_shape = (640, 480)  # Postavi veličinu slike (može se prilagoditi prema kamerama)

# Kalibracija za bočnu kameru
mtx_side, dist_side, rvecs_side, tvecs_side, error_side = calibrate_camera(objpoints_side, imgpoints_side, image_shape)

# 2. Prikupljanje uzoraka i kalibracija za gornju kameru (top camera)
print("Prikupljanje uzoraka za gornju kameru (top). Pritisnite 's' za spremanje uzorka.")
objpoints_top, imgpoints_top = collect_samples(camera_id=0, num_samples=10,
                                               camera_name="Gornja kamera (top)")  # Gornja kamera

# Kalibracija za gornju kameru
mtx_top, dist_top, rvecs_top, tvecs_top, error_top = calibrate_camera(objpoints_top, imgpoints_top, image_shape)

# 3. Stereo kalibracija s 10 uzoraka i pre-estimated parametrima
print("Prikupljanje uzoraka za stereo kalibraciju. Pritisnite 's' za spremanje uzorka.")
objpoints_stereo, imgpoints1_stereo, imgpoints2_stereo = [], [], []
cap_side = cv2.VideoCapture(1)
cap_top = cv2.VideoCapture(0)

while len(objpoints_stereo) < 10:
    ret_side, frame_side = cap_side.read()
    ret_top, frame_top = cap_top.read()

    if not ret_side or not ret_top:
        print("Jedna od kamera nije dostupna!")
        break

    gray_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)
    gray_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)

    ret_side, corners_side = cv2.findChessboardCorners(gray_side, chessboard_size, None)
    ret_top, corners_top = cv2.findChessboardCorners(gray_top, chessboard_size, None)

    if ret_side and ret_top:
        cv2.drawChessboardCorners(frame_side, chessboard_size, corners_side, ret_side)
        cv2.drawChessboardCorners(frame_top, chessboard_size, corners_top, ret_top)

        cv2.imshow('Bočna kamera (side)', frame_side)
        cv2.imshow('Gornja kamera (top)', frame_top)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            imgpoints1_stereo.append(corners_side)
            imgpoints2_stereo.append(corners_top)
            objpoints_stereo.append(objp)
            print(f"Uzeti stereo uzorak {len(objpoints_stereo)}")

            # Stereo kalibracija i ispis greške nakon svakog uzorka
            if len(objpoints_stereo) >= 3:  # Minimalno 3 uzorka potrebno za stereo kalibraciju
                R, T, E, F, stereo_error = stereo_calibration_pre_estimated(
                    objpoints_stereo, imgpoints1_stereo, imgpoints2_stereo, mtx_side, dist_side, mtx_top, dist_top, image_shape
                )
                print(f"Stereo reprojekcijska greška nakon {len(objpoints_stereo)} uzoraka: {stereo_error}")

    else:
        cv2.imshow('Bočna kamera (side)', frame_side)
        cv2.imshow('Gornja kamera (top)', frame_top)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_side.release()
cap_top.release()
cv2.destroyAllWindows()

# Spremanje kalibracijskih rezultata nakon prikupljanja 10 uzoraka
R, T, E, F, stereo_error = stereo_calibration_pre_estimated(
    objpoints_stereo, imgpoints1_stereo, imgpoints2_stereo, mtx_side, dist_side, mtx_top, dist_top, image_shape
)

# 4. Spremanje rezultata u JSON
calibration_data = {
    "side_camera": {
        "matrix": mtx_side.tolist(),
        "distortion": dist_side.tolist(),
    },
    "top_camera": {
        "matrix": mtx_top.tolist(),
        "distortion": dist_top.tolist(),
    },
    "stereo": {
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "error": stereo_error
    }
}

with open("calibration_data.json", "w") as json_file:
    json.dump(calibration_data, json_file, indent=4)

print("Kalibracijski podaci su spremljeni u 'calibration_data.json'.")
