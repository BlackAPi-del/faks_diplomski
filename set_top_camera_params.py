import cv2
import json

# Inicijalizacija gornje kamere
cap = cv2.VideoCapture(0)  # 1 označava gornju kameru (prilagodi prema tvojoj postavci)

# Očitavanje trenutnih postavki s kamere
autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
focus_value = cap.get(cv2.CAP_PROP_FOCUS)

# Ispis trenutnih postavki
print(f"Trenutna vrijednost autofokusa: {autofocus}")
print(f"Trenutna vrijednost automatske ekspozicije: {auto_exposure}")
print(f"Trenutna vrijednost fokusa: {focus_value}")

# Snimanje slike za pregled postavki
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prikaz frame-a
    cv2.imshow("Top Camera - Press 's' to save settings", frame)

    # Kontrola fokusa putem tipkovnice
    key = cv2.waitKey(1) & 0xFF

    if key == ord('+'):  # Povećaj fokus
        focus_value += 1
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        print(f"Fokus povećan: {focus_value}")

    elif key == ord('-'):  # Smanji fokus
        focus_value -= 1
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        print(f"Fokus smanjen: {focus_value}")

    # Pritisni 's' za spremanje postavki
    elif key == ord('s'):
        # Spremanje postavki u JSON datoteku
        settings = {
            "autofocus": cap.get(cv2.CAP_PROP_AUTOFOCUS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "focus": focus_value
        }

        with open('top_camera_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        print("Postavke za gornju kameru su spremljene.")

    # Pritisni 'q' za izlaz
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
