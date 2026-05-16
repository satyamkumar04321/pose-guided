import cv2
import mediapipe as mp
import numpy as np
import math


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

            # Draw landmark point
            cv2.circle(output_frame, (x, y), 6, (0, 255, 0), -1)

    # ==========================
    # VALIDATE POSE
    # ==========================

    valid_pose = False
    validation_message = "No body detected"

    if landmarks:
        valid_pose, validation_message = self.is_valid_pose(landmarks)

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
            (left_shoulder_y + right_shoulder_y) / 2
        )

        # Move slightly downward
        shoulder_y += 20

        # --------------------------
        # HIP Y
        # --------------------------

        left_hip_y = landmarks["LEFT_HIP"]["y"]
        right_hip_y = landmarks["RIGHT_HIP"]["y"]

        hip_y = int(
            (left_hip_y + right_hip_y) / 2
        )

        # --------------------------
        # WAIST Y
        # --------------------------

        waist_y = int(
            (shoulder_y + hip_y) / 2
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
                f"Shoulder: {shoulder_width}",
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
                f"Waist: {waist_width}",
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
        "message": validation_message,
        "shoulder_width": shoulder_width,
        "waist_width": waist_width,
        "body_ratio": body_ratio
    }