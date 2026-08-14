import os
import sys
import glob
import yaml
import numpy as np
import cv2 as cv
from scipy import linalg
calibration_settings = {}
def extract_high_precision_corners(
    image_gray, pattern_size, win_size=(7, 7)
):
    # 1. Enhance contrast locally using CLAHE
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(image_gray)

    # 2. Initial corner detection
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(
        enhanced_gray, pattern_size, flags
    )

    if not found:
        return False, None

    # 3. High-precision sub-pixel refinement with expanded iteration limit
    subpix_criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        300,
        1e-6,
    )

    # Dead-zone (-1, -1) avoids self-matching adjacent pixels
    corners_refined = cv.cornerSubPix(
        enhanced_gray, corners, win_size, (-1, -1), subpix_criteria
    )

    return True, corners_refined
def DLT(P1, P2, point1, point2):
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print("File does not exist:", filename)
        quit()

    print("Using for calibration settings: ", filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if "camera0" not in calibration_settings.keys():
        print(
            "camera0 key was not found in the settings file. Check calibration_settings.yaml"
        )
        quit()
def save_frames_single_camera(camera_name):
    if not os.path.exists("frames"):
        os.mkdir("frames")

    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]

    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No video data received. Exiting...")
            quit()

        frame_small = cv.resize(
            frame, None, fx=1 / view_resize, fy=1 / view_resize
        )

        if not start:
            cv.putText(
                frame_small,
                "Press SPACEBAR to start collection frames",
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv.putText(
                frame_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            if cooldown <= 0:
                savename = os.path.join(
                    "frames", camera_name + "_" + str(saved_count) + ".png"
                )
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow("frame_small", frame_small)
        k = cv.waitKey(1)

        if k == 27:
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    images_names = sorted(glob.glob(images_prefix))
    images = [cv.imread(imname, 1) for imname in images_names]

    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 1e-5)

    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # Original working 3D object points
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    height, width = images[0].shape[:2]

    imgpoints = []
    objpoints = []

    for frame in images:
        gray = frame[:, :, 1]
        ret, corners = extract_high_precision_corners(gray, (rows, columns))

        if ret:
            corners = cv.cornerSubPix(
                gray, corners, (3, 3), (-1, -1), subpix_criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners)

    mono_flags = (
    cv.CALIB_RATIONAL_MODEL  # Enables k4, k5, k6
    + cv.CALIB_THIN_PRISM_MODEL  # Enables s1, s2, s3, s4
    + cv.CALIB_ZERO_TANGENT_DIST  # Locks p1, p2 if optics are centered
    )
    solver_criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    500,
    1e-6,
    )
# Run for both camera0 and camera1 mono intrinsics
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        (width, height),
        None,
        None,
        criteria=solver_criteria,
        flags=mono_flags,
    )
    print(f"Intrinsic RMSE ({images_prefix}):", ret)

    return cmtx, dist
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    out_filename = os.path.join(
        "camera_parameters", camera_name + "_intrinsics.dat"
    )
    with open(out_filename, "w") as outf:
        outf.write("intrinsic:\n")
        for l in camera_matrix:
            for en in l:
                outf.write(str(en) + " ")
            outf.write("\n")

        outf.write("distortion:\n")
        for en in distortion_coefs[0]:
            outf.write(str(en) + " ")
        outf.write("\n")
def save_frames_two_cams(camera0_name, camera1_name):
    if not os.path.exists("frames_pair"):
        os.mkdir("frames_pair")

    number_to_save = calibration_settings["stereo_calibration_frames"]

    aa = 1
    saved_count = 0
    while True:
        frame0 = cv.imread(f"l_image{aa:03d}.png", 1)
        frame1 = cv.imread(f"r_image{aa:03d}.png", 1)

        if frame0 is None or frame1 is None:
            print(f"Finished loading frames up to index {aa-1}.")
            break

        savename0 = os.path.join(
            "frames_pair", camera0_name + "_" + str(saved_count) + ".png"
        )
        cv.imwrite(savename0, frame0)

        savename1 = os.path.join(
            "frames_pair", camera1_name + "_" + str(saved_count) + ".png"
        )
        cv.imwrite(savename1, frame1)

        saved_count += 1
        aa += 1

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
def stereo_calibrate(
    mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1
):
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 1e-5)

    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = frame0[:, :, 1]
        gray2 = frame1[:, :, 1]
        c_ret1, corners1 = extract_high_precision_corners(gray1, (rows, columns))
        c_ret2, corners2 = extract_high_precision_corners(gray2, (rows, columns))

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(
                gray1, corners1, (3, 3), (-1, -1), subpix_criteria
            )
            corners2 = cv.cornerSubPix(
                gray2, corners2, (3, 3), (-1, -1), subpix_criteria
            )

            # --- FIX: Ensure Left and Right frames start indexing from the SAME physical corner ---
            # Distance check between Point 0 in Left and Point 0 in Right
            dist_p0 = np.linalg.norm(corners1[0, 0] - corners2[0, 0])
            dist_p0_flipped = np.linalg.norm(corners1[0, 0] - corners2[-1, 0])

            # If the flipped point is closer, the right camera detected the board in reverse order
            if dist_p0_flipped < dist_p0:
                corners2 = corners2[::-1]

            # Draw visual markers
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)

            cv.imshow("Left Frame", frame0)
            cv.imshow("Right Frame", frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord("s"):
                print("Skipping bad stereo pair")
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()

    # --- FIX: CALIB_USE_INTRINSIC_GUESS prevents focal errors from corrupting Tz ---
    stereocalibration_flags = (
    cv.CALIB_USE_INTRINSIC_GUESS
    + cv.CALIB_SAME_FOCAL_LENGTH
    + cv.CALIB_FIX_PRINCIPAL_POINT
    + cv.CALIB_RATIONAL_MODEL
    )
    stereo_criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    500,
    1e-6,
)
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=stereo_criteria,
        flags=stereocalibration_flags,
    )

    print("Stereo RMSE: ", ret)
    print("Refined Left K:\n", CM1)
    print("Refined Right K:\n", CM2)
    print("Rotation R:\n", R)
    print("Translation T:\n", T)

    return CM1, dist0, CM2, dist1, R, T, imgpoints_left, imgpoints_right
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P
def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=""):
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    camera0_filename = os.path.join(
        "camera_parameters", prefix + "camera0_rot_trans.dat"
    )
    with open(camera0_filename, "w") as outf:
        outf.write("R:\n")
        for l in R0:
            for en in l:
                outf.write(str(en) + " ")
            outf.write("\n")
        outf.write("T:\n")
        for l in T0:
            for en in l:
                outf.write(str(en) + " ")
            outf.write("\n")

    camera1_filename = os.path.join(
        "camera_parameters", prefix + "camera1_rot_trans.dat"
    )
    with open(camera1_filename, "w") as outf:
        outf.write("R:\n")
        for l in R1:
            for en in l:
                outf.write(str(en) + " ")
            outf.write("\n")
        outf.write("T:\n")
        for l in T1:
            for en in l:
                outf.write(str(en) + " ")
            outf.write("\n")

    return R0, T0, R1, T1
if __name__ == "__main__":
    parse_calibration_settings_file(
        r"C:\Users\kljke\OneDrive\Documentos\Staj\ZERK\stereo\trial\trial\images\calibration_settings.yaml"
    )

    # Step 2: Intrinsics
    images_prefix_l = os.path.join("l_*")
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix_l)
    save_camera_intrinsics(cmtx0, dist0, "camera0")

    images_prefix_r = os.path.join("r_*")
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix_r)
    save_camera_intrinsics(cmtx1, dist1, "camera1")

    # Step 3: Copy frames to frames_pair
    save_frames_two_cams("camera0", "camera1")

    # Step 4: Stereo Calibration
    frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
    frames_prefix_c1 = os.path.join("frames_pair", "camera1*")

    (
        CM1,
        dist0,
        CM2,
        dist1,
        R,
        T,
        imgpoints_left,
        imgpoints_right,
    ) = stereo_calibrate(
        cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
    )

    # Step 5: Save extrinsics
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T)

    # Save to NPZ with properly formatted numpy arrays
    np.savez(
        "calibration_data_v2.npz",
        K0_calib=CM1,
        K1_calib=CM2,
        dist0_calib=dist0,
        dist1_calib=dist1,
        R_calib=R,
        T_calib=T,
        pts_left=np.array(imgpoints_left, dtype=np.float32),
        pts_right=np.array(imgpoints_right, dtype=np.float32),
    )
