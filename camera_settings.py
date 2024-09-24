import json
import cv2


def load_camera_settings(filename):
    with open(filename, 'r') as file:
        settings = json.load(file)
    return settings


def apply_camera_settings(camera, settings):
    # Postavljanje autofokusa
    camera.set(cv2.CAP_PROP_AUTOFOCUS, settings['autofocus'])

    # Postavljanje automatske ekspozicije
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings['auto_exposure'])

    # Postavljanje fokusa
    camera.set(cv2.CAP_PROP_FOCUS, settings['focus'])
