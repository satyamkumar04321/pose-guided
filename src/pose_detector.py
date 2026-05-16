import cv2
import mediapipe as mp
import numpy as np
import math


class PoseDetector:

    def __init__(self):

        # ==========================
        # MEDIAPIPE MODULES
        # ==========================

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation

        # ==========================
        # POSE MODEL
        # ==========================

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ==========================
        # SEGMENTATION MODEL
        # ==========================

        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

        # ==========================
        # IMPORTANT LANDMARKS
        # ==========================

        self.landmark_ids = {
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28
        }

        # ==========================
        # STABILITY TRACKING
        # ==========================

        self.previous_center = None

    # ==========================================
    # DISTANCE FUNCTION
    # ==========================================

    def calculate_distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ==========================================
    # POSE VALIDATION
    # ==========================================

    def is_valid_pose(self, landmarks):

        required = [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_ANKLE",
            "RIGHT_ANKLE"
        ]

        # ==========================
        # CHECK ALL LANDMARKS EXIST
        # ==========================

        for point in required:

            if point not in landmarks:
                return False, "Full body not detected"

            if landmarks[point]["visibility"] < 0.7:
                return False, "Body not clearly visible"

        # ==========================
        # SHOULDER ALIGNMENT
        # ==========================

        left_shoulder_y = landmarks["LEFT_SHOULDER"]["y"]
        right_shoulder_y = landmarks["RIGHT_SHOULDER"]["y"]

        shoulder_diff = abs(left_shoulder_y - right_shoulder_y)

        if shoulder_diff > 30:
            return False, "Stand straight"

        # ==========================
        # HIP ALIGNMENT
        # ==========================

        left_hip_y = landmarks["LEFT_HIP"]["y"]
        right_hip_y = landmarks["RIGHT_HIP"]["y"]

        hip_diff = abs(left_hip_y - right_hip_y)

        if hip_diff > 35:
            return False, "Face camera properly"

        # ==========================
        # BODY STABILITY
        # ==========================

        shoulder_center = (
            (
                landmarks["LEFT_SHOULDER"]["x"] +
                landmarks["RIGHT_SHOULDER"]["x"]
            ) // 2,

            (
                landmarks["LEFT_SHOULDER"]["y"] +
                landmarks["RIGHT_SHOULDER"]["y"]
            ) // 2
        )

        if self.previous_center is not None:

            movement = self.calculate_distance(
                shoulder_center,
                self.previous_center
            )

            if movement > 20:
                self.previous_center = shoulder_center
                return False, "Stay still"

        self.previous_center = shoulder_center

        return True, "Perfect"

    # ==========================================
    # MAIN DETECTION FUNCTION
    # ==========================================

    def detect_pose(self, frame):

        output_frame = frame.copy()

        # ==========================
        # RGB CONVERSION
        # ==========================

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ==========================
        # POSE DETECTION
        # ==========================

        pose_results = self.pose.process(rgb_frame)

        # ==========================
        # SEGMENTATION
        # ==========================

        segmentation_results = self.selfie_segmentation.process(rgb_frame)

        # ==========================
        # IMAGE DIMENSIONS
        # ==========================

        height, width, _ = frame.shape

        # ==========================
        # SEGMENTATION MASK
        # ==========================

        segmentation_mask = segmentation_results.segmentation_mask

        condition = segmentation_mask > 0.5

        binary_mask = np.zeros((height, width), dtype=np.uint8)

        binary_mask[condition] = 255

        # ==========================
        # LANDMARK STORAGE
        # ==========================

        landmarks = {}

        # ==========================
        # EXTRACT LANDMARKS
        # ==========================

        if pose_results.pose_landmarks:

            self.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            for name, idx in self.landmark_ids.items():

                landmark = pose_results.pose_landmarks.landmark[idx]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmarks[name] = {
                    "x": x,
                    "y": y,
                    "visibility": landmark.visibility
                }

                # Draw point
                cv2.circle(output_frame, (x, y), 6, (0, 255, 0), -1)

                # Draw label
                cv2.putText(
                    output_frame,
                    name,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # ==========================
        # VALIDATE POSE
        # ==========================

        valid_pose = False
        validation_message = "No body detected"

        if landmarks:
            valid_pose, validation_message = self.is_valid_pose(landmarks)

        # ==========================
        # VALIDATION STATUS DISPLAY
        # ==========================

        color = (0, 255, 0) if valid_pose else (0, 0, 255)

        cv2.putText(
            output_frame,
            validation_message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return {
            "frame": output_frame,
            "mask": binary_mask,
            "landmarks": landmarks,
            "valid_pose": valid_pose,
            "message": validation_message
        }