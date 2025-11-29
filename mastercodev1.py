import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys

"/mnt/d/KALMAN ADAT/BME/KUTATOMUNKA/PyCharm/motorcycle_mpc/ray_save/PPO_PathTrackEnv_200_spielberg_v4_seed01"

VIDEOINPUT = "motoros_video_5.mp4"
VIDEOOUTPUT = "distancetracker_v0.mp4"
MODEL_PATH = "yolov8x-seg.pt"
MASK_CLASSES = [0, 3]
TRACK_CLASS = 3

CONFIDENCE_THRESHOLD = 0.72

calibration_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Pont: {x}, {y}")


# Homográfia távolság számoló
def get_dist_homography(pts_src, point, real_w, real_l):
    pts_src = np.array(pts_src, dtype=np.float32)
    pts_dst = np.array([
        [0, 0],
        [real_w, 0],
        [real_w, real_l],
        [0, real_l]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    vec = np.array([[[point[0], point[1]]]], dtype=np.float32)
    res = cv2.perspectiveTransform(vec, matrix)
    return res[0][0][1]

print("--- 1. FÁZIS: MOTOR KERESÉSE ---")
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
video_cap = cv2.VideoCapture(VIDEOINPUT)
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))
out = cv2.VideoWriter(VIDEOOUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

cv2.namedWindow("System")
cv2.setMouseCallback("System", mouse_callback)

detected_motor_box = None
frozen_frame = None

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Hiba: Videó vége motor nélkül.")
        sys.exit()

    results = model(frame, device=device, verbose=False)[0]

    found = False
    for data in results.boxes.data.tolist():
        if float(data[4]) > CONFIDENCE_THRESHOLD and int(data[5]) == TRACK_CLASS:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            # Méret szűrés (zaj ellen)
            if (xmax - xmin) * (ymax - ymin) > 5000:
                detected_motor_box = (xmin, ymin, xmax, ymax)
                frozen_frame = frame.copy()
                found = True
                break

    if found:
        print(">>> MOTOR MEGTALÁLVA! FAGYASZTÁS. <<<")
        break

    cv2.imshow("System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit()

print("\n--- 2. FÁZIS: JELÖLÉS ---")
print("1. Kattints 4 pontot a képen (Trapéz)")
print("2. Nyomj SPACE-t")

reference_data = {}

while True:
    display_frame = frozen_frame.copy()

    mx, my, mMx, mMy = detected_motor_box
    cv2.rectangle(display_frame, (mx, my), (mMx, mMy), (255, 0, 0), 2)
    cv2.putText(display_frame, "MOTOR (REF)", (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for pt in calibration_points:
        cv2.circle(display_frame, pt, 6, (0, 0, 255), -1)

    if len(calibration_points) == 4:
        cv2.polylines(display_frame, [np.array(calibration_points)], True, (0, 255, 0), 2)
        cv2.putText(display_frame, "KESZ! NYOMJ SPACE-T", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, f"Pontok: {len(calibration_points)}/4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    cv2.imshow("System", display_frame)
    key = cv2.waitKey(100)

    if len(calibration_points) == 4 and key != -1:
        cv2.destroyWindow("System")

        print("\n" + "=" * 40)
        try:
            user_w = float(input("Kijelölt SZÉLESSÉG (méter): "))
            user_l = float(input("Kijelölt HOSSZ (méter): "))
        except ValueError:
            print("Hibás adat! Default: 6m x 20m")
            user_w, user_l = 6.0, 20.0
        print("=" * 40 + "\n")

        w_ref = mMx - mx
        h_ref = mMy - my
        area_ref = w_ref * h_ref
        motor_bottom = ((mx + mMx) // 2, mMy)

        ref_dist = get_dist_homography(calibration_points, motor_bottom, user_w, user_l)
        reference_data["const"] = ref_dist * np.sqrt(area_ref)

        cv2.namedWindow("System")
        break

    if key == ord('q'): sys.exit()

print("--- 3. FÁZIS: MÉRÉS ---")
tracker = DeepSort(max_age=70, n_init=3)

while True:
    ret, frame = video_cap.read()
    if not ret: break

    results = model(frame, device=device, verbose=False, retina_masks=True)[0]

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for i, mask in enumerate(masks):
            cid = int(classes[i])
            if cid in MASK_CLASSES:
                color = (0, 255, 0) if cid == 3 else (0, 165, 255)
                m_res = cv2.resize(mask, (frame_width, frame_height))
                overlay[m_res > 0.5] = color

        frame = cv2.addWeighted(frame, 1, overlay, 0.4, 0)

    detections = []
    for data in results.boxes.data.tolist():
        if float(data[4]) < 0.3: continue
        cid = int(data[5])

        if cid == TRACK_CLASS:
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            w, h = x2 - x1, y2 - y1
            detections.append([[x1, y1, w, h], data[4], cid])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed(): continue

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        w_curr = x2 - x1
        h_curr = y2 - y1
        area_curr = w_curr * h_curr

        dist = 0
        if area_curr > 0:
            dist = reference_data["const"] / np.sqrt(area_curr)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{dist:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("System", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

video_cap.release()
out.release()
cv2.destroyAllWindows()
print("Kész.")
