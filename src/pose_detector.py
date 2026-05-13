import cv2
import mediapipe as mp


class PoseDetector:

    def __init__(self):

        # MediaPipe Pose module
        self.mp_pose = mp.solutions.pose

        # Drawing utility
        self.mp_drawing = mp.solutions.drawing_utils

        # Pose model initialization
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Important landmarks for V1
        self.landmark_ids = {
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28
        }

    def detect_pose(self, frame):

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose
        results = self.pose.process(rgb_frame)

        # Dictionary for storing landmarks
        landmarks = {}

        # Get image dimensions
        height, width, _ = frame.shape

        # Check if pose landmarks detected
        if results.pose_landmarks:

            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Extract selected landmarks
            for name, idx in self.landmark_ids.items():

                landmark = results.pose_landmarks.landmark[idx]

                # Convert normalized coordinates to pixels
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Store coordinates
                landmarks[name] = (x, y)

                # Draw point for debugging
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Show landmark name
                cv2.putText(
                    frame,
                    name,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return frame, landmarks