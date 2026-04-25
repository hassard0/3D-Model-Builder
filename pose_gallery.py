"""OpenPose-format pose gallery generator.

HunyuanDiT-ControlNet-Pose was trained on OpenPose-18 skeleton images: colored
lines for bones, colored circles for joints, black background. We define a
set of canonical poses as 18-keypoint configurations and render them at 1024×1024.

Keypoint indices (COCO / OpenPose-18):
  0 Nose    1 Neck    2 R-Sh   3 R-El   4 R-Wr   5 L-Sh   6 L-El   7 L-Wr
  8 R-Hip   9 R-Kn   10 R-An  11 L-Hip 12 L-Kn  13 L-An  14 R-Eye 15 L-Eye
 16 R-Ear  17 L-Ear
"""
from __future__ import annotations
import os
from PIL import Image, ImageDraw
import math

# OpenPose default limb connections + colors (RGB).
LIMBS = [
    (1, 2),   # neck -> right shoulder
    (1, 5),   # neck -> left shoulder
    (2, 3),   # right upper arm
    (3, 4),   # right forearm
    (5, 6),   # left upper arm
    (6, 7),   # left forearm
    (1, 8),   # neck -> right hip
    (8, 9),   # right thigh
    (9, 10),  # right shin
    (1, 11),  # neck -> left hip
    (11, 12), # left thigh
    (12, 13), # left shin
    (1, 0),   # neck -> nose
    (0, 14),  # nose -> right eye
    (14, 16), # right eye -> ear
    (0, 15),  # nose -> left eye
    (15, 17), # left eye -> ear
]
LIMB_COLORS = [
    (255, 0, 0),   (255, 85, 0),  (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0),  (0, 255, 0),   (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255),   (85, 0, 255),  (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
JOINT_COLORS = LIMB_COLORS + [(255, 0, 85)]  # 18 joints


def _p(x, y): return (float(x), float(y))


# Canonical T-pose layout. Other poses are expressed as delta rotations of
# this baseline, but for clarity/debuggability we just redeclare each pose
# as explicit keypoints.
def tpose():
    return [
        _p(0.500, 0.16),  # 0 nose
        _p(0.500, 0.24),  # 1 neck
        _p(0.430, 0.26),  # 2 R shoulder
        _p(0.300, 0.26),  # 3 R elbow
        _p(0.175, 0.26),  # 4 R wrist
        _p(0.570, 0.26),  # 5 L shoulder
        _p(0.700, 0.26),  # 6 L elbow
        _p(0.825, 0.26),  # 7 L wrist
        _p(0.455, 0.54),  # 8 R hip
        _p(0.455, 0.76),  # 9 R knee
        _p(0.455, 0.96),  # 10 R ankle
        _p(0.545, 0.54),  # 11 L hip
        _p(0.545, 0.76),  # 12 L knee
        _p(0.545, 0.96),  # 13 L ankle
        _p(0.480, 0.145), # 14 R eye
        _p(0.520, 0.145), # 15 L eye
        _p(0.460, 0.155), # 16 R ear
        _p(0.540, 0.155), # 17 L ear
    ]


def apose():
    kp = tpose()
    # arms slightly down-and-out
    kp[2] = _p(0.438, 0.275)
    kp[3] = _p(0.355, 0.38)
    kp[4] = _p(0.280, 0.47)
    kp[5] = _p(0.562, 0.275)
    kp[6] = _p(0.645, 0.38)
    kp[7] = _p(0.720, 0.47)
    return kp


def idle():
    kp = tpose()
    # arms hanging down at sides
    kp[2] = _p(0.438, 0.27); kp[3] = _p(0.44, 0.42); kp[4] = _p(0.44, 0.56)
    kp[5] = _p(0.562, 0.27); kp[6] = _p(0.56, 0.42); kp[7] = _p(0.56, 0.56)
    return kp


def walking():
    kp = idle()
    # right leg forward, left leg back. right arm back, left arm forward.
    kp[8] = _p(0.46, 0.54); kp[9] = _p(0.40, 0.72); kp[10] = _p(0.36, 0.94)
    kp[11] = _p(0.54, 0.54); kp[12] = _p(0.60, 0.72); kp[13] = _p(0.64, 0.94)
    kp[2] = _p(0.438, 0.27); kp[3] = _p(0.48, 0.42); kp[4] = _p(0.53, 0.54)
    kp[5] = _p(0.562, 0.27); kp[6] = _p(0.52, 0.42); kp[7] = _p(0.47, 0.54)
    return kp


def running():
    kp = idle()
    # deeper swing + bent arms
    kp[8] = _p(0.46, 0.54); kp[9] = _p(0.34, 0.70); kp[10] = _p(0.28, 0.88)
    kp[11] = _p(0.54, 0.54); kp[12] = _p(0.66, 0.70); kp[13] = _p(0.72, 0.88)
    kp[2] = _p(0.43, 0.27); kp[3] = _p(0.55, 0.34); kp[4] = _p(0.48, 0.44)
    kp[5] = _p(0.57, 0.27); kp[6] = _p(0.45, 0.34); kp[7] = _p(0.52, 0.44)
    return kp


def sitting():
    kp = idle()
    # hips at center, thighs horizontal, knees at 90°, torso upright
    kp[1] = _p(0.500, 0.40)  # neck lower
    kp[0] = _p(0.500, 0.30)
    kp[14] = _p(0.48, 0.28); kp[15] = _p(0.52, 0.28)
    kp[16] = _p(0.46, 0.29); kp[17] = _p(0.54, 0.29)
    kp[2] = _p(0.44, 0.43); kp[5] = _p(0.56, 0.43)
    kp[3] = _p(0.41, 0.57); kp[6] = _p(0.59, 0.57)
    kp[4] = _p(0.40, 0.68); kp[7] = _p(0.60, 0.68)
    kp[8] = _p(0.46, 0.70); kp[11] = _p(0.54, 0.70)
    kp[9] = _p(0.40, 0.70); kp[12] = _p(0.60, 0.70)  # knees at 90° (out in front)
    kp[10] = _p(0.40, 0.92); kp[13] = _p(0.60, 0.92)
    return kp


def crouch():
    kp = idle()
    # compressed vertical, knees bent
    kp[1] = _p(0.500, 0.42)
    kp[0] = _p(0.500, 0.32)
    kp[14] = _p(0.48, 0.30); kp[15] = _p(0.52, 0.30)
    kp[16] = _p(0.46, 0.31); kp[17] = _p(0.54, 0.31)
    kp[2] = _p(0.44, 0.45); kp[5] = _p(0.56, 0.45)
    kp[3] = _p(0.36, 0.56); kp[6] = _p(0.64, 0.56)
    kp[4] = _p(0.32, 0.66); kp[7] = _p(0.68, 0.66)
    kp[8] = _p(0.46, 0.65); kp[11] = _p(0.54, 0.65)
    kp[9] = _p(0.40, 0.78); kp[12] = _p(0.60, 0.78)
    kp[10] = _p(0.42, 0.93); kp[13] = _p(0.58, 0.93)
    return kp


def wave_right():
    kp = idle()
    # Right arm raised high, hand waving
    kp[2] = _p(0.44, 0.26); kp[3] = _p(0.35, 0.15); kp[4] = _p(0.30, 0.04)
    return kp


def jump():
    kp = idle()
    # both arms up, knees tucked
    kp[1] = _p(0.500, 0.28)
    kp[0] = _p(0.500, 0.18)
    kp[14] = _p(0.48, 0.165); kp[15] = _p(0.52, 0.165)
    kp[16] = _p(0.46, 0.175); kp[17] = _p(0.54, 0.175)
    kp[2] = _p(0.44, 0.30); kp[5] = _p(0.56, 0.30)
    kp[3] = _p(0.40, 0.18); kp[6] = _p(0.60, 0.18)
    kp[4] = _p(0.38, 0.06); kp[7] = _p(0.62, 0.06)
    kp[8] = _p(0.46, 0.58); kp[11] = _p(0.54, 0.58)
    kp[9] = _p(0.40, 0.70); kp[12] = _p(0.60, 0.70)
    kp[10] = _p(0.36, 0.80); kp[13] = _p(0.64, 0.80)
    return kp


def power_pose():
    # hands on hips, legs wide
    kp = tpose()
    kp[2] = _p(0.44, 0.275); kp[5] = _p(0.56, 0.275)
    kp[3] = _p(0.40, 0.40);  kp[6] = _p(0.60, 0.40)
    kp[4] = _p(0.46, 0.52);  kp[7] = _p(0.54, 0.52)
    kp[8] = _p(0.44, 0.54);  kp[11] = _p(0.56, 0.54)
    kp[9] = _p(0.40, 0.76);  kp[12] = _p(0.60, 0.76)
    kp[10] = _p(0.36, 0.96); kp[13] = _p(0.64, 0.96)
    return kp


def victory():
    # both arms in V above head
    kp = tpose()
    kp[2] = _p(0.44, 0.27); kp[5] = _p(0.56, 0.27)
    kp[3] = _p(0.36, 0.17); kp[6] = _p(0.64, 0.17)
    kp[4] = _p(0.28, 0.05); kp[7] = _p(0.72, 0.05)
    return kp


def fighting_stance():
    kp = idle()
    # boxer's guard, right foot forward
    kp[2] = _p(0.43, 0.28); kp[3] = _p(0.40, 0.40); kp[4] = _p(0.44, 0.48)
    kp[5] = _p(0.57, 0.28); kp[6] = _p(0.60, 0.40); kp[7] = _p(0.56, 0.48)
    kp[8] = _p(0.46, 0.55);  kp[11] = _p(0.54, 0.55)
    kp[9] = _p(0.40, 0.74);  kp[12] = _p(0.60, 0.74)
    kp[10] = _p(0.36, 0.92); kp[13] = _p(0.60, 0.92)
    return kp


def kneeling():
    kp = idle()
    # one knee on ground, other knee bent up
    kp[1] = _p(0.500, 0.32)
    kp[0] = _p(0.500, 0.22); kp[14] = _p(0.48, 0.20); kp[15] = _p(0.52, 0.20)
    kp[16] = _p(0.46, 0.215); kp[17] = _p(0.54, 0.215)
    kp[2] = _p(0.44, 0.35); kp[5] = _p(0.56, 0.35)
    kp[3] = _p(0.42, 0.50); kp[6] = _p(0.58, 0.50)
    kp[4] = _p(0.42, 0.64); kp[7] = _p(0.58, 0.64)
    kp[8] = _p(0.46, 0.60); kp[11] = _p(0.54, 0.60)
    # Right leg back kneeling
    kp[9] = _p(0.44, 0.82); kp[10] = _p(0.50, 0.92)
    # Left leg bent up
    kp[12] = _p(0.60, 0.72); kp[13] = _p(0.56, 0.92)
    return kp


def ballet():
    # arms up-out, one leg extended back
    kp = tpose()
    kp[2] = _p(0.44, 0.27); kp[3] = _p(0.36, 0.18); kp[4] = _p(0.28, 0.10)
    kp[5] = _p(0.56, 0.27); kp[6] = _p(0.64, 0.18); kp[7] = _p(0.72, 0.10)
    kp[8] = _p(0.46, 0.55); kp[11] = _p(0.54, 0.55)
    kp[9] = _p(0.40, 0.70); kp[10] = _p(0.36, 0.90)
    kp[12] = _p(0.66, 0.68); kp[13] = _p(0.80, 0.70)
    return kp


def dab():
    # one arm extended up-right, other tucked toward it
    kp = idle()
    kp[5] = _p(0.56, 0.27); kp[6] = _p(0.72, 0.18); kp[7] = _p(0.88, 0.08)
    kp[2] = _p(0.44, 0.27); kp[3] = _p(0.56, 0.20); kp[4] = _p(0.64, 0.14)
    kp[0] = _p(0.45, 0.19); kp[1] = _p(0.50, 0.26)
    return kp


POSES = {
    "tpose":       ("T-pose",        tpose),
    "apose":       ("A-pose",        apose),
    "idle":        ("Idle",          idle),
    "walking":     ("Walking",       walking),
    "running":     ("Running",       running),
    "sitting":     ("Sitting",       sitting),
    "crouch":      ("Crouching",     crouch),
    "wave":        ("Wave right",    wave_right),
    "jump":        ("Jump",          jump),
    "power":       ("Power pose",    power_pose),
    "victory":     ("Victory",       victory),
    "fight":       ("Fighting",      fighting_stance),
    "kneel":       ("Kneeling",      kneeling),
    "ballet":      ("Ballet",        ballet),
    "dab":         ("Dab",           dab),
}


def render_pose(keypoints: list, size: int = 1024, line_width: int = 14, joint_radius: int = 10) -> Image.Image:
    """Render an OpenPose-18 keypoint list as a skeleton image on black."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = [(x * size, y * size) for (x, y) in keypoints]

    # Draw limbs first so joints render on top
    for (a, b), color in zip(LIMBS, LIMB_COLORS):
        if a < len(pts) and b < len(pts):
            draw.line([pts[a], pts[b]], fill=color, width=line_width)

    # Draw joints
    for i, (x, y) in enumerate(pts):
        color = JOINT_COLORS[i % len(JOINT_COLORS)]
        draw.ellipse(
            (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
            fill=color, outline=None,
        )
    return img


def build_gallery(out_dir: str, size: int = 1024):
    """Render every pose in POSES to <out_dir>/<key>.png and return a manifest."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for key, (label, fn) in POSES.items():
        img = render_pose(fn(), size=size)
        img.save(os.path.join(out_dir, f"{key}.png"))
        manifest.append({"key": key, "label": label, "url": f"{key}.png"})
    return manifest


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "poses"
    m = build_gallery(out)
    print(f"rendered {len(m)} poses to {out}/")
    for item in m:
        print(f"  {item['key']}: {item['label']}")
