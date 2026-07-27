"""
sanity_check_fix.py — run this AFTER swapping in the new baseline.py.

verify_review.py was written to run against the OLD code (it compares the
current sign against a flipped one). This script instead calls the real,
patched find_best_antipodal_pair directly and checks it now behaves.

No dataset needed.

    python sanity_check_fix.py
"""
import numpy as np
import cv2

from baseline import (
    estimate_normals, get_largest_contour, find_best_antipodal_pair,
    pair_to_grasp_rectangle, xywh_theta_to_corners, enforce_min_dist,
)
from data_loading import convert


def make_shape(name):
    m = np.zeros((480, 640), np.uint8)
    if name == "ellipse":
        cv2.ellipse(m, (320, 240), (80, 40), 20, 0, 360, 255, -1)
    elif name == "rect":
        cv2.rectangle(m, (260, 200), (380, 270), 255, -1)
    elif name == "L":
        cv2.rectangle(m, (260, 180), (320, 300), 255, -1)
        cv2.rectangle(m, (260, 250), (400, 300), 255, -1)
    elif name == "ring":
        cv2.circle(m, (320, 240), 70, 255, -1)
        cv2.circle(m, (320, 240), 35, 0, -1)
    return m


# expected grasp width: the narrow dimension the gripper should close across
EXPECTED = {"ellipse": 80.0, "rect": 70.0, "L": 60.0, "ring": None}

print("=" * 72)
print("B1 — antipodal score after the fix (a perfect pair scores 2.00)")
print("=" * 72)

all_ok = True
for name in ["ellipse", "rect", "L", "ring"]:
    contour = get_largest_contour(make_shape(name))
    normals = estimate_normals(contour)

    centroid = contour.astype(float).mean(axis=0)
    outward = sum(1 for p, v in zip(contour, normals)
                  if np.dot(v, p.astype(float) - centroid) > 0)

    pair, score = find_best_antipodal_pair(contour, normals, 19, 90)
    width = np.linalg.norm(pair[1] - pair[0])

    exp = EXPECTED[name]
    ok = score > 1.2 and (exp is None or abs(width - exp) < 12)
    all_ok &= ok

    print(f"  {name:8s} outward normals {outward}/{len(contour):<5} "
          f"score {score:+.3f}/2.00   width {width:6.1f} px"
          + (f"  (expected ~{exp:.0f})" if exp else "")
          + ("   OK" if ok else "   <-- UNEXPECTED"))

print()
print("=" * 72)
print("B2 — min_dist is clamped so no prediction is narrower than the plate")
print("=" * 72)
for min_d, plate in [(19, 30), (25, 30), (30, 30), (35, 30), (40, 26)]:
    eff = enforce_min_dist(min_d, plate)
    print(f"  min_dist={min_d:3d}, plate_thickness={plate:3d}  ->  effective min_dist={eff:3d}"
          + ("   (clamped)" if eff != min_d else ""))

print("\n  round-trip check — does convert() report the intended angle?")
for w in [20, 29, 30, 45]:
    corners = xywh_theta_to_corners(300.0, 200.0, w, 30.0, 10.0)
    *_, theta = convert(corners)
    flag = "  <-- 90 deg flip" if abs(theta - 10.0) > 45 else "  ok"
    print(f"    width {w:3d} px vs plate 30 px, intended 10.0 deg -> {theta:6.1f} deg{flag}")
print("    (with min_dist clamped to plate_thickness, widths below the plate")
print("     can no longer be produced, so the flip cases are unreachable)")

print()
print("PASS — B1 fix is live." if all_ok else
      "FAIL — check that baseline.py was actually replaced.")