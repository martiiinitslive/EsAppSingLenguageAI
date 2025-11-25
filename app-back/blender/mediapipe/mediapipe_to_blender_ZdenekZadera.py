import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the input file
input_file = Path("poses_mediapipe.json")
with open(input_file, "r", encoding="utf-8") as f:
    mediapipe_data = json.load(f)

# Define finger joint structure for MediaPipe (21 landmarks)
# Thumb: [1,2,3,4], Index: [5,6,7,8], Middle: [9,10,11,12], Ring: [13,14,15,16], Pinky: [17,18,19,20]
finger_joints = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

# Function to compute rotation between two vectors
def compute_rotation(vec_from: np.ndarray, vec_to: np.ndarray) -> Dict[str, List[float]]:
    # Normalize vectors
    v1 = vec_from / np.linalg.norm(vec_from)
    v2 = vec_to / np.linalg.norm(vec_to)

    # Compute rotation matrix using cross product and dot product
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if np.isclose(dot, 1.0):
        rot_matrix = np.eye(3)
    else:
        skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
        rot_matrix = np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))

    # Convert to quaternion and Euler angles
    rotation = R.from_matrix(rot_matrix)
    quaternion = rotation.as_quat().tolist()  # [x, y, z, w]
    euler = rotation.as_euler('XYZ', degrees=True).tolist()

    return {"quaternion": quaternion, "euler": euler}

# Prepare output structure
output_data = {"meta": {"converted_from": "mediapipe", "converted_to": "blender", "source_count": mediapipe_data["meta"]["source_count"]}, "poses": {}}

# Iterate over all gestures
for gesture_name, gesture_info in mediapipe_data["poses"].items():
    if not gesture_info.get("hand_detected", False):
        continue

    world_landmarks = gesture_info.get("world_landmarks", [])
    if len(world_landmarks) != 21:
        continue

    # Convert landmarks to numpy array
    points = np.array([[lm["x"], lm["y"], lm["z"]] for lm in world_landmarks])

    # Compute rotations for each finger
    finger_rotations = {}
    for finger, indices in finger_joints.items():
        rotations = []
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            vec_from = np.array([1.0, 0.0, 0.0])  # Blender default bone direction (X-axis)
            vec_to = points[end_idx] - points[start_idx]
            rotation_data = compute_rotation(vec_from, vec_to)
            rotations.append(rotation_data)
        finger_rotations[finger] = rotations

    # Add to output
    output_data["poses"][gesture_name] = {
        "gesture_name": gesture_name,
        "handedness": gesture_info.get("handedness", "Unknown"),
        "rotations": finger_rotations
    }

# Save output file
output_file = Path("poses_library.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

print(f"Conversion completed. Output saved to {output_file} with {len(output_data['poses'])} gestures.")