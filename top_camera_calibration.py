import cv2
import numpy as np
import os
import json


class TopCameraCalibrator:
    def __init__(self, checkerboard_size=(10, 7), square_size=0.025):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.objpoints = []  # 3D točke u stvarnom prostoru
        self.imgpoints = []  # 2D točke u slici
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.samples_taken = 0

    def load_camera_params(self, filepath):
        """Učitavanje postavki kamere iz JSON datoteke."""
        with open(filepath, 'r') as f:
            settings = json.load(f)
        print(f"Postavke kamere učitane iz {filepath}: {settings}")

    def take_sample(self, frame):
        """Uzmi uzorak slike i pronađi kutove checkerboard uzorka."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        if ret:
            # Pripremi objektne točke
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size

            # Refine corners for more accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), None)

            # Dodaj točke
            self.objpoints.append(objp)
            self.imgpoints.append(corners2)

            # Povećaj broj uzoraka
            self.samples_taken += 1

            # Iscrtavanje kutova za vizualnu povratnu informaciju
            cv2.drawChessboardCorners(frame, self.checkerboard_size, corners2, ret)
            print(f"Uzorak {self.samples_taken}/15 uzet.")
        else:
            print("Checkerboard nije pronađen. Provjerite osvjetljenje, udaljenost, ili fokus kamere.")

    def calibrate(self, frame_size):
        """Kalibriraj kameru i izračunaj reprojekcijsku pogrešku."""
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, frame_size, None, None)

        if ret:
            print("Kalibracija uspješna!")
            # Izračunaj reprojekcijsku grešku
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i],
                                                  self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            mean_error /= len(self.objpoints)
            print(f"Reprojekcijska pogreška: {mean_error}")

            # Spremi kalibracijske podatke u datoteku (JSON)
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
            }
            self.save_calibration_data(calibration_data)
        else:
            print("Kalibracija nije uspjela.")

    def save_calibration_data(self, data):
        """Spremi podatke o kalibraciji u JSON datoteku."""
        calibration_file = 'calibration_data/top_camera_calibration.json'
        os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
        with open(calibration_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Kalibracijski podaci spremljeni u {calibration_file}")


# Glavna funkcija za uzimanje uzoraka i kalibraciju
def main():
    calibrator = TopCameraCalibrator()

    # Učitaj parametre kamere
    calibrator.load_camera_params('top_camera_settings.json')

    cap = cv2.VideoCapture(0)
    while calibrator.samples_taken < 15:
        ret, frame = cap.read()
        if not ret:
            print("Greška pri čitanju iz kamere.")
            break

        # Prikaz streama s iscrtanim linijama checkerboard uzorka
        cv2.imshow('Top Camera Calibration', frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            calibrator.take_sample(frame)

        if key == ord('q'):
            print("Prekinuto od strane korisnika.")
            break

    if calibrator.samples_taken == 15:
        print("Uzeto 15 uzoraka, započinjem kalibraciju...")
        calibrator.calibrate(frame.shape[1::-1])  # Proslijedi samo širinu i visinu

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
