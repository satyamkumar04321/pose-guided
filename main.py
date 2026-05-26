import cv2
import time
import os
import glob

from src.pose_detector import PoseDetector


# =====================================
# LOAD POSE RECOMMENDATIONS
# =====================================

POSE_FOLDERS = {
    "Oval": "Poses/oval",
    "Triangle": "Poses/triangle",
    "Rectangle": "Poses/rectangle",
    "Trapezium": "Poses/trapezium",
    "Inverted Triangle": "Poses/inverted_triangle"
}


# =====================================
# LOAD RECOMMENDED IMAGES
# =====================================

def load_pose_images(body_type):

    folder = POSE_FOLDERS.get(body_type)

    if folder is None:
        return []

    image_paths = glob.glob(
        os.path.join(folder, "*.jpg")
    )

    return image_paths


# =====================================
# DISPLAY RECOMMENDATIONS
# =====================================

def show_pose_recommendations(body_type):

    image_paths = load_pose_images(body_type)

    if len(image_paths) == 0:

        print(f"No pose images found for {body_type}")

        return

    for index, image_path in enumerate(image_paths):

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (400, 700))

        cv2.putText(
            image,
            body_type,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            f"Recommended Pose {index + 1}",
            image
        )


# =====================================
# MAIN FUNCTION
# =====================================

def main():

    # =====================================
    # INITIALIZE CAMERA
    # =====================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Cannot access webcam")

        return

    # =====================================
    # INITIALIZE DETECTOR
    # =====================================

    detector = PoseDetector()

    # =====================================
    # AUTO CAPTURE VARIABLES
    # =====================================

    valid_start_time = None

    capture_delay = 3

    captured = False

    frozen_result = None

    # =====================================
    # MAIN LOOP
    # =====================================

    while True:

        # =====================================
        # LIVE MODE
        # =====================================

        if not captured:

            ret, frame = cap.read()

            if not ret:
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            # =====================================
            # DETECT POSE
            # =====================================

            result = detector.detect_pose(frame)

            output_frame = result["frame"]

            # =====================================
            # VALID POSE
            # =====================================

            if result["valid_pose"]:

                # Start timer
                if valid_start_time is None:
                    valid_start_time = time.time()

                elapsed_time = (
                    time.time() - valid_start_time
                )

                remaining_time = max(
                    0,
                    int(capture_delay - elapsed_time) + 1
                )

                # Countdown
                cv2.putText(
                    output_frame,
                    f"Capturing in {remaining_time}",
                    (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                # =====================================
                # AUTO CAPTURE
                # =====================================

                if elapsed_time >= capture_delay:

                    frozen_result = result

                    captured = True

                    # =====================================
                    # SHOW RECOMMENDATIONS
                    # =====================================

                    body_type = result["body_type"]

                    show_pose_recommendations(
                        body_type
                    )

            else:

                # Reset timer
                valid_start_time = None

            # =====================================
            # SHOW LIVE FRAME
            # =====================================

            cv2.imshow(
                "PoseGuide",
                output_frame
            )

        # =====================================
        # CAPTURED MODE
        # =====================================

        else:

            frozen_frame = (
                frozen_result["frame"].copy()
            )

            cv2.putText(
                frozen_frame,
                "CAPTURED",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            cv2.imshow(
                "PoseGuide",
                frozen_frame
            )

            cv2.imshow(
                "Segmentation Mask",
                frozen_result["mask"]
            )

        # =====================================
        # KEYBOARD CONTROLS
        # =====================================

        key = cv2.waitKey(1)

        # RESET
        if key == ord('r'):

            captured = False

            frozen_result = None

            valid_start_time = None

            cv2.destroyAllWindows()

        # QUIT
        elif key == ord('q'):
            break

    # =====================================
    # CLEANUP
    # =====================================

    cap.release()

    cv2.destroyAllWindows()


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":

    main()