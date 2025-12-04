import numpy as np
import cv2
import os
import glob
import sys

BASE_PATH = r"D:\KITTI_MINI"

print(f"--- KITTI VALIDATOR INDÍTÁSA ---")
print(f"Mappa: {BASE_PATH}")

img_dir = r"C:\Users\szeleskalman\Desktop\archive\2011_09_26_drive_0001_sync\image_02\data"
velo_dir = r"C:\Users\szeleskalman\Desktop\archive\2011_09_26_drive_0001_sync\velodyne_points\data"
calib_dir = r"C:\Users\szeleskalman\Desktop\archive\data_object_calib\testing\calib"

if not os.path.exists(img_dir) and os.path.exists(os.path.join(BASE_PATH, "training")):
    BASE_PATH = os.path.join(BASE_PATH, "training")
    img_dir = os.path.join(BASE_PATH, "image_2")
    velo_dir = os.path.join(BASE_PATH, "velodyne")
    calib_dir = os.path.join(BASE_PATH, "calib")

img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
velo_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.txt")))

def read_calib(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()]).reshape(-1)
    return data


def project_lidar(velo, calib):
    P2 = calib['P2'].reshape(3, 4)
    R0 = calib['R0_rect'].reshape(3, 3)
    Tr = calib['Tr_velo_to_cam'].reshape(3, 4)

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = R0

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = Tr

    velo = velo[velo[:, 0] > 0]
    velo_hom = np.hstack((velo[:, :3], np.ones((velo.shape[0], 1))))

    cam_points = np.dot(velo_hom, Tr_velo_to_cam.T)
    rect_points = np.dot(cam_points, R0_rect.T)
    img_points_hom = np.dot(rect_points, P2.T)

    img_points = np.zeros((img_points_hom.shape[0], 2))
    img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
    img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]

    depths = rect_points[:, 2]
    return img_points, depths

cv2.namedWindow("KITTI Ground Truth")
mouse_x, mouse_y = 0, 0


def mouse_move(event, x, y):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


cv2.setMouseCallback("KITTI Ground Truth", mouse_move)

for i in range(len(img_files)):
    frame = cv2.imread(img_files[i])
    h, w, _ = frame.shape

    velo_data = np.fromfile(velo_files[i], dtype=np.float32).reshape(-1, 4)
    calib_data = read_calib(calib_files[i])
    pts_2d, depths = project_lidar(velo_data, calib_data)

    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    pts_2d = pts_2d[mask]
    depths = depths[mask]

    while True:
        display = frame.copy()

        closest_pt = None
        min_dist_to_mouse = 20.0
        closest_depth_val = 0

        for j in range(0, len(pts_2d), 3):
            pt = pts_2d[j]
            d = depths[j]

            dist_to_mouse = np.sqrt((pt[0] - mouse_x) ** 2 + (pt[1] - mouse_y) ** 2)

            if dist_to_mouse < min_dist_to_mouse:
                min_dist_to_mouse = dist_to_mouse
                closest_pt = pt
                closest_depth_val = d

            c_val = min(255, int(d * 3))
            color = (255 - c_val, 0, c_val)
            cv2.circle(display, (int(pt[0]), int(pt[1])), 2, color, -1)

        if closest_pt is not None:
            cv2.circle(display, (int(closest_pt[0]), int(closest_pt[1])), 8, (255, 255, 255), -1)

            label = f"{closest_depth_val:.2f} m"
            cv2.putText(display, label, (int(closest_pt[0]) + 15, int(closest_pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Piros betű

            cv2.putText(display, f"GROUND TRUTH: {closest_depth_val:.2f} m", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("KITTI Ground Truth", display)

        key = cv2.waitKey(10)
        if key == 32: break
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()

cv2.destroyAllWindows()
