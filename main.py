import cv2
from src.pose_detector import PoseDetector


def main():

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check webcam
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    # Initialize detector
    detector = PoseDetector()

    while True:

        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read frame")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect pose
        output_frame, landmarks = detector.detect_pose(frame)

        # Print landmarks
        if landmarks:
            print(landmarks)

        # Show frame
        cv2.imshow("PoseGuide - Pose Detection", output_frame)

        # Exit on Q key
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()