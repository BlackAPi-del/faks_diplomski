import json

def load_calibration_data(filename):
    with open(filename, 'r') as file:
        calibration_data = json.load(file)
    return calibration_data
