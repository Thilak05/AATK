import cv2
import numpy as np
from pyzbar import pyzbar
import json
from datetime import datetime

def calculate_age(dob):
    birth_date = datetime.strptime(dob, "%d-%m-%Y")
    today = datetime.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def scan_qr_code():
    cap = cv2.VideoCapture(0)
    qr_code_data = None

    while True:
        ret, frame = cap.read()

        decoded_objects = pyzbar.decode(frame)

        for obj in decoded_objects:
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array(points, dtype=np.float32))
                points = list(map(tuple, np.squeeze(hull)))

            n = len(points)
            for j in range(0, n):
                cv2.line(frame, points[j], points[(j + 1) % n], (255, 0, 0), 3)

            qr_code_data = obj.data.decode("utf-8")
            print("QR Code Data:", qr_code_data)

            data = json.loads(qr_code_data)

            
            cap.release()
            cv2.destroyAllWindows()
            return data 

        cv2.imshow('QR Code Scanner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# This part will not be executed when imported into app.py
if __name__ == "__main__":
    scan_qr_code()
