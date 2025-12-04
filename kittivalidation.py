import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os
import glob

IMG_DIR = r"C:\Users\szeleskalman\Desktop\archive\2011_09_26_drive_0001_sync\image_02\data"

MODEL_PATH = "yolov8x-seg.pt"
TRACK_CLASS = 2
CONFIDENCE_THRESHOLD = 0.50
MIN_AREA_THRESHOLD = 5000

device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30, n_init=3)
calibration_points = []
reference_data = {}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Pont: {x}, {y}")


def get_dist_homography(pts_src, point, real_w, real_l):
    pts_src = np.array(pts_src, dtype=np.float32)
    pts_dst = np.array([[0, 0], [real_w, 0], [real_w, real_l], [0, real_l]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    vec = np.array([[[point[0], point[1]]]], dtype=np.float32)
    res = cv2.perspectiveTransform(vec, matrix)
    return res[0][0][1]

print("\n--- 1. FÁZIS: MEGFELELŐ AUTÓ KERESÉSE ---")
print("A rendszer keres egy jól látható autót a képeken...")

found_ref_car = None
frozen_frame = None
start_index = 0

for i, img_path in enumerate(img_files):
    frame = cv2.imread(img_path)
    if frame is None: continue

    results = model(frame, device=device, verbose=False)[0]

    best_area = 0
    best_box = None

    for data in results.boxes.data.tolist():
        if float(data[4]) < CONFIDENCE_THRESHOLD: continue
        if int(data[5]) == TRACK_CLASS:
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            area = (x2 - x1) * (y2 - y1)

            if area > MIN_AREA_THRESHOLD and area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is not None:
        found_ref_car = best_box
        frozen_frame = frame.copy()
        start_index = i
        print(f"✅ REFERENCIA AUTÓ MEGTALÁLVA A {i}. KÉPEN!")
        print(f"   Doboz: {best_box}, Terület: {best_area}")
        break

    cv2.imshow("Kereses...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit()

if frozen_frame is None:
    print("❌ Nem találtam megfelelő méretű autót a képeken!")
    sys.exit()

print("\n--- 2. FÁZIS: ÚT KALIBRÁCIÓ ---")
print("1. Jelöld ki a sávot (4 pont) a 'frozen' képen.")
print("2. Nyomj SPACE-t a rögzítéshez.")

cv2.namedWindow("Estimator")
cv2.setMouseCallback("Estimator", mouse_callback)

while True:
    disp = frozen_frame.copy()

    rx1, ry1, rx2, ry2 = found_ref_car
    cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
    cv2.putText(disp, "AUTO REFERENCIA", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    for pt in calibration_points:
        cv2.circle(disp, pt, 5, (0, 255, 0), -1)

    if len(calibration_points) == 4:
        cv2.polylines(disp, [np.array(calibration_points)], True, (0, 255, 0), 2)
        cv2.putText(disp, "KESZ! NYOMJ SPACE-T", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(disp, f"Jelolj 4 pontot az uton: {len(calibration_points)}/4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow("Estimator", disp)
    key = cv2.waitKey(20)

    if key == 32 and len(calibration_points) == 4:
        cv2.destroyWindow("Estimator")
        try:
            print("\n" + "=" * 40)
            user_w = float(input("SZÉLESSÉG (méter, pl. 3.0): "))
            user_l = float(input("HOSSZ (méter, pl. 10.0): "))
        except ValueError:
            user_w, user_l = 3.0, 10.0
            print("Hibás adat, alapértelmezett (3m x 10m) használata.")
        print("=" * 40 + "\n")
      
        ref_bottom_center = ((rx1 + rx2) // 2, ry2)
        ref_dist = get_dist_homography(calibration_points, ref_bottom_center, user_w, user_l)

        ref_area = (rx2 - rx1) * (ry2 - ry1)

        reference_data["const"] = ref_dist * np.sqrt(ref_area)
        break

    if key == ord('q'): sys.exit()

print("\n--- 3. FÁZIS: FUTTATÁS ---")
print("Újraindítom a videót/képeket az elejéről...")

tracker = DeepSort(max_age=30, n_init=3)
cv2.namedWindow("Final Result")

for i, img_path in enumerate(img_files):
    frame = cv2.imread(img_path)
    if frame is None: continue

    results = model(frame, device=device, verbose=False, retina_masks=True)[0]

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        overlay = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for j, mask in enumerate(masks):
            if int(classes[j]) == TRACK_CLASS:
                m_res = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                overlay[m_res > 0.5] = (0, 255, 0)
        frame = cv2.addWeighted(frame, 1, overlay, 0.4, 0)

    detections = []
    for data in results.boxes.data.tolist():
        if float(data[4]) < CONFIDENCE_THRESHOLD: continue
        if int(data[5]) == TRACK_CLASS:
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            w, h = x2 - x1, y2 - y1
            detections.append([[x1, y1, w, h], data[4], int(data[5])])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed(): continue
        if track.time_since_update > 1: continue

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        w_curr = x2 - x1
        h_curr = y2 - y1
        area_curr = w_curr * h_curr

        dist_meters = 0
        if area_curr > 0:
            dist_meters = reference_data["const"] / np.sqrt(area_curr)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"En: {dist_meters:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Final Result", frame)

    key = cv2.waitKey(0)
    if key == 32: continue
    if key == ord('q'): break

cv2.destroyAllWindows()
