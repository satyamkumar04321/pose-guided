import cv2
import mediapipe as mp
import numpy as np
import math


class PoseDetector:

    def __init__(self):

        # =====================================
        # MEDIAPIPE MODULES
        # =====================================

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation

        # =====================================
        # POSE MODEL
        # =====================================

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # =====================================
        # SEGMENTATION MODEL
        # =====================================

        self.selfie_segmentation = (
            self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        )

        # =====================================
        # IMPORTANT LANDMARKS
        # =====================================

        self.landmark_ids = {
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28
        }

        # =====================================
        # BODY TYPE PROTOTYPES
        # =====================================

        self.body_profiles = {
            "Oval": 0.70,
            "Triangle": 0.85,
            "Rectangle": 1.00,
            "Trapezium": 1.12,
            "Inverted Triangle": 1.30
        }

        # =====================================
        # STABILITY TRACKING
        # =====================================

        self.previous_center = None

    # =====================================
    # DISTANCE FUNCTION
    # =====================================

    def calculate_distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # =====================================
    # BODY WIDTH EXTRACTION
    # =====================================

    def get_body_width(self, mask, y, center_x):

        height, width = mask.shape

        # Safety checks
        if y < 0 or y >= height:
            return None, None, None

        if center_x < 0 or center_x >= width:
            return None, None, None

        row = mask[y]

        # Center must belong to body
        if row[center_x] != 255:
            return None, None, None

        # ==========================
        # SCAN LEFT
        # ==========================

        left_x = center_x

        while left_x > 0 and row[left_x] == 255:
            left_x -= 1

        # ==========================
        # SCAN RIGHT
        # ==========================

        right_x = center_x

        while right_x < width - 1 and row[right_x] == 255:
            right_x += 1

        body_width = right_x - left_x

        return left_x, right_x, body_width

    # =====================================
    # BODY CLASSIFICATION
    # =====================================

    def classify_body_type(self, body_ratio):

        if body_ratio is None:
            return "Unknown", 0

        closest_type = None

        smallest_distance = float("inf")

        for body_type, ideal_ratio in self.body_profiles.items():

            distance = abs(body_ratio - ideal_ratio)

            if distance < smallest_distance:

                smallest_distance = distance
                closest_type = body_type

        confidence = max(
            0,
            round((1 - smallest_distance) * 100)
        )

        return closest_type, confidence

    # =====================================
    # POSE VALIDATION
    # =====================================

    def is_valid_pose(self, landmarks):

        required = [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_ANKLE",
            "RIGHT_ANKLE"
        ]

        for point in required:

            if point not in landmarks:
                return False, "Full body not detected"

            if landmarks[point]["visibility"] < 0.7:
                return False, "Body not clearly visible"

        # =====================================
        # SHOULDER ALIGNMENT
        # =====================================

        shoulder_diff = abs(
            landmarks["LEFT_SHOULDER"]["y"] -
            landmarks["RIGHT_SHOULDER"]["y"]
        )

        if shoulder_diff > 30:
            return False, "Stand straight"

        # =====================================
        # HIP ALIGNMENT
        # =====================================

        hip_diff = abs(
            landmarks["LEFT_HIP"]["y"] -
            landmarks["RIGHT_HIP"]["y"]
        )

        if hip_diff > 35:
            return False, "Face camera properly"

        # =====================================
        # STABILITY CHECK
        # =====================================

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

    # =====================================
    # MAIN DETECTION FUNCTION
    # =====================================

    def detect_pose(self, frame):

        output_frame = frame.copy()

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =====================================
        # POSE DETECTION
        # =====================================

        pose_results = self.pose.process(rgb_frame)

        # =====================================
        # SEGMENTATION
        # =====================================

        segmentation_results = (
            self.selfie_segmentation.process(rgb_frame)
        )

        height, width, _ = frame.shape

        # =====================================
        # SEGMENTATION MASK
        # =====================================

        segmentation_mask = (
            segmentation_results.segmentation_mask
        )

        condition = segmentation_mask > 0.5

        binary_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        binary_mask[condition] = 255

        # =====================================
        # LANDMARK STORAGE
        # =====================================

        landmarks = {}

        # =====================================
        # BODY VARIABLES
        # =====================================

        shoulder_width = None
        waist_width = None
        body_ratio = None
        body_type = "Unknown"
        confidence = 0

        # =====================================
        # EXTRACT LANDMARKS
        # =====================================

        if pose_results.pose_landmarks:

            self.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            for name, idx in self.landmark_ids.items():

                landmark = (
                    pose_results.pose_landmarks.landmark[idx]
                )

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmarks[name] = {
                    "x": x,
                    "y": y,
                    "visibility": landmark.visibility
                }

                cv2.circle(
                    output_frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

        # =====================================
        # VALIDATE POSE
        # =====================================

        valid_pose = False
        validation_message = "No body detected"

        if landmarks:

            valid_pose, validation_message = (
                self.is_valid_pose(landmarks)
            )

        # =====================================
        # WIDTH EXTRACTION
        # =====================================

        if valid_pose:

            # =====================================
            # SHOULDER HEIGHT
            # =====================================

            shoulder_y = int(
                (
                    landmarks["LEFT_SHOULDER"]["y"] +
                    landmarks["RIGHT_SHOULDER"]["y"]
                ) / 2
            )

            shoulder_y += 20

            # =====================================
            # HIP HEIGHT
            # =====================================

            hip_y = int(
                (
                    landmarks["LEFT_HIP"]["y"] +
                    landmarks["RIGHT_HIP"]["y"]
                ) / 2
            )

            # =====================================
            # WAIST HEIGHT
            # =====================================

            waist_y = int(
                shoulder_y +
                ((hip_y - shoulder_y) * 0.35)
            )

            # =====================================
            # BODY CENTER
            # =====================================

            center_x = int(
                (
                    landmarks["LEFT_SHOULDER"]["x"] +
                    landmarks["RIGHT_SHOULDER"]["x"]
                ) / 2
            )

            # =====================================
            # SHOULDER WIDTH
            # =====================================

            s_left, s_right, shoulder_width = (
                self.get_body_width(
                    binary_mask,
                    shoulder_y,
                    center_x
                )
            )

            # =====================================
            # WAIST WIDTH
            # =====================================

            w_left, w_right, waist_width = (
                self.get_body_width(
                    binary_mask,
                    waist_y,
                    center_x
                )
            )

            # =====================================
            # DRAW SHOULDER LINE
            # =====================================

            if shoulder_width is not None:

                cv2.line(
                    output_frame,
                    (s_left, shoulder_y),
                    (s_right, shoulder_y),
                    (0, 255, 255),
                    3
                )

            # =====================================
            # DRAW WAIST LINE
            # =====================================

            if waist_width is not None:

                cv2.line(
                    output_frame,
                    (w_left, waist_y),
                    (w_right, waist_y),
                    (255, 0, 255),
                    3
                )

            # =====================================
            # BODY RATIO
            # =====================================

            if shoulder_width and waist_width:

                body_ratio = (
                    shoulder_width / waist_width
                )

                body_type, confidence = (
                    self.classify_body_type(body_ratio)
                )

        # =====================================
        # DISPLAY OUTPUT
        # =====================================

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

        if body_ratio:

            cv2.putText(
                output_frame,
                f"Ratio: {body_ratio:.2f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Body Type: {body_type}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                f"Confidence: {confidence}%",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
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
            "body_ratio": body_ratio,
            "body_type": body_type,
            "confidence": confidence
        }