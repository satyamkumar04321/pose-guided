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
    # WIDTH EXTRACTION FUNCTION
    # ==========================================

    def get_body_width(self, mask, y):

        # Prevent invalid indexing
        if y < 0 or y >= mask.shape[0]:
            return None, None, None

        # Get row pixels
        row = mask[y]

        # Find white pixels
        white_pixels = np.where(row == 255)[0]

        # No body detected
        if len(white_pixels) == 0:
            return None, None, None

        # Left boundary
        left_x = white_pixels[0]

        # Right boundary
        right_x = white_pixels[-1]

        # Width
        width = right_x - left_x

        return left_x, right_x, width

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
        # CHECK LANDMARK VISIBILITY
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

        shoulder_diff = abs(
            left_shoulder_y - right_shoulder_y
        )

        if shoulder_diff > 30:
            return False, "Stand straight"

        # ==========================
        # HIP ALIGNMENT
        # ==========================

        left_hip_y = landmarks["LEFT_HIP"]["y"]
        right_hip_y = landmarks["RIGHT_HIP"]["y"]

        hip_diff = abs(
            left_hip_y - right_hip_y
        )

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

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ==========================
        # POSE DETECTION
        # ==========================

        pose_results = self.pose.process(rgb_frame)

        # ==========================
        # SEGMENTATION
        # ==========================

        segmentation_results = self.selfie_segmentation.process(
            rgb_frame
        )

        # ==========================
        # IMAGE DIMENSIONS
        # ==========================

        height, width, _ = frame.shape

        # ==========================
        # SEGMENTATION MASK
        # ==========================

        segmentation_mask = segmentation_results.segmentation_mask

        condition = segmentation_mask > 0.5

        binary_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        binary_mask[condition] = 255

        # ==========================
        # LANDMARK STORAGE
        # ==========================

        landmarks = {}

        # ==========================
        # WIDTH VARIABLES
        # ==========================

        shoulder_width = None
        waist_width = None
        body_ratio = None

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

                # Draw points
                cv2.circle(
                    output_frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

        # ==========================
        # VALIDATE POSE
        # ==========================

        valid_pose = False
        validation_message = "No body detected"

        if landmarks:
            valid_pose, validation_message = self.is_valid_pose(
                landmarks
            )

        # ==========================
        # WIDTH EXTRACTION
        # ==========================

        if valid_pose:

            # --------------------------
            # SHOULDER Y
            # --------------------------

            left_shoulder_y = landmarks["LEFT_SHOULDER"]["y"]
            right_shoulder_y = landmarks["RIGHT_SHOULDER"]["y"]

            shoulder_y = int(
                (
                    left_shoulder_y +
                    right_shoulder_y
                ) / 2
            )

            # Move downward slightly
            shoulder_y += 20

            # --------------------------
            # HIP Y
            # --------------------------

            left_hip_y = landmarks["LEFT_HIP"]["y"]
            right_hip_y = landmarks["RIGHT_HIP"]["y"]

            hip_y = int(
                (
                    left_hip_y +
                    right_hip_y
                ) / 2
            )

            # --------------------------
            # WAIST Y
            # --------------------------

            waist_y = int(
                (
                    shoulder_y +
                    hip_y
                ) / 2
            )

            # ==========================
            # SHOULDER WIDTH
            # ==========================

            s_left, s_right, shoulder_width = self.get_body_width(
                binary_mask,
                shoulder_y
            )

            # ==========================
            # WAIST WIDTH
            # ==========================

            w_left, w_right, waist_width = self.get_body_width(
                binary_mask,
                waist_y
            )

            # ==========================
            # DRAW SHOULDER LINE
            # ==========================

            if shoulder_width is not None:

                cv2.line(
                    output_frame,
                    (s_left, shoulder_y),
                    (s_right, shoulder_y),
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    output_frame,
                    f"Shoulder Width: {shoulder_width}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # ==========================
            # DRAW WAIST LINE
            # ==========================

            if waist_width is not None:

                cv2.line(
                    output_frame,
                    (w_left, waist_y),
                    (w_right, waist_y),
                    (255, 0, 255),
                    3
                )

                cv2.putText(
                    output_frame,
                    f"Waist Width: {waist_width}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2
                )

            # ==========================
            # BODY RATIO
            # ==========================

            if shoulder_width and waist_width:

                body_ratio = shoulder_width / waist_width

                cv2.putText(
                    output_frame,
                    f"Ratio: {body_ratio:.2f}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

        # ==========================
        # VALIDATION MESSAGE
        # ==========================

        color = (
            (0, 255, 0)
            if valid_pose
            else (0, 0, 255)
        )

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
            "message": validation_message,
            "shoulder_width": shoulder_width,
            "waist_width": waist_width,
            "body_ratio": body_ratio
        }