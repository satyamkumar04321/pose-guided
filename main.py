import cv2
import time

from src.pose_detector import PoseDetector


def main():

    # ==========================
    # INITIALIZE WEBCAM
    # ==========================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return

    # ==========================
    # INITIALIZE DETECTOR
    # ==========================

    detector = PoseDetector()

    # ==========================
    # AUTO CAPTURE VARIABLES
    # ==========================

    valid_start_time = None

    capture_delay = 3

    captured = False

    frozen_result = None

    # ==========================
    # MAIN LOOP
    # ==========================

    while True:

        # ==========================
        # LIVE MODE
        # ==========================

        if not captured:

            ret, frame = cap.read()

            if not ret:
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            # Detect pose
            result = detector.detect_pose(frame)

            output_frame = result["frame"]

            # ==========================
            # VALID POSE DETECTED
            # ==========================

            if result["valid_pose"]:

                # Start timer
                if valid_start_time is None:
                    valid_start_time = time.time()

                elapsed_time = time.time() - valid_start_time

                remaining_time = max(
                    0,
                    int(capture_delay - elapsed_time) + 1
                )

                # Countdown display
                cv2.putText(
                    output_frame,
                    f"Capturing in {remaining_time}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                # AUTO CAPTURE
                if elapsed_time >= capture_delay:

                    frozen_result = result

                    captured = True

            # ==========================
            # INVALID POSE
            # ==========================

            else:

                valid_start_time = None

            # Show live frame
            cv2.imshow("PoseGuide", output_frame)

        # ==========================
        # CAPTURED MODE
        # ==========================

        else:

            frozen_frame = frozen_result["frame"]

            cv2.putText(
                frozen_frame,
                "CAPTURED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            cv2.imshow("PoseGuide", frozen_frame)

            cv2.imshow(
                "Segmentation Mask",
                frozen_result["mask"]
            )

        # ==========================
        # KEYBOARD CONTROLS
        # ==========================

        key = cv2.waitKey(1)

        # R → Reset
        if key == ord('r'):

            captured = False

            frozen_result = None

            valid_start_time = None

        # Q → Quit
        elif key == ord('q'):
            break

    # ==========================
    # CLEANUP
    # ==========================

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()