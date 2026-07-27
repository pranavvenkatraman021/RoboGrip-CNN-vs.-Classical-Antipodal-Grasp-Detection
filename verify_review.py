"""
verify_review.py — run this from inside the repo folder BEFORE changing anything.

It re-derives the two main claims in the review using YOUR functions,
so you don't have to take an external review's word for it.
No dataset needed; it builds synthetic shapes.

    python verify_review.py
"""
import numpy as np
import cv2

from baseline import estimate_normals, get_largest_contour, xywh_theta_to_corners
from data_loading import convert


# ----------------------------------------------------------------------
# CHECK 1: do the contour normals point outward, and does the antipodal
#          score therefore reward the wrong pairs?
# ----------------------------------------------------------------------
def make_shape(name):
    m = np.zeros((480, 640), np.uint8)
    if name == "ellipse":
        cv2.ellipse(m, (320, 240), (80, 40), 20, 0, 360, 255, -1)
    elif name == "rect":
        cv2.rectangle(m, (260, 200), (380, 270), 255, -1)
    elif name == "L":
        cv2.rectangle(m, (260, 180), (320, 300), 255, -1)
        cv2.rectangle(m, (260, 250), (400, 300), 255, -1)
    elif name == "ring":  # mug-like
        cv2.circle(m, (320, 240), 70, 255, -1)
        cv2.circle(m, (320, 240), 35, 0, -1)
    return m


def search(contour, normals, sign, min_dist=19, max_dist=90, sample_step=4):
    n = len(contour)
    idx = range(0, n, sample_step)
    best = (-np.inf, None)
    for i in idx:
        for j in idx:
            if i >= j:
                continue
            pa, pb = contour[i].astype(float), contour[j].astype(float)
            d = np.linalg.norm(pb - pa)
            if d < min_dist or d > max_dist:
                continue
            dab = (pb - pa) / d
            s = sign * (np.dot(normals[i], dab) + np.dot(normals[j], -dab))
            if s > best[0]:
                best = (s, (pa, pb))
    return best


print("=" * 74)
print("CHECK 1 — normal orientation and antipodal score sign")
print("=" * 74)
print(f"{'shape':9s} {'outward normals':>16s} {'current best':>13s} {'width':>7s} "
      f"{'sign-flipped':>13s} {'width':>7s}")
for name in ["ellipse", "rect", "L", "ring"]:
    c = get_largest_contour(make_shape(name))
    nrm = estimate_normals(c)
    cen = c.mean(axis=0)
    outward = sum(1 for p, v in zip(c, nrm) if np.dot(v, p.astype(float) - cen) > 0)
    s_cur, (a1, b1) = search(c, nrm, +1)
    s_fix, (a2, b2) = search(c, nrm, -1)
    print(f"{name:9s} {outward}/{len(c):<12} {s_cur:+13.3f} {np.linalg.norm(b1-a1):7.1f} "
          f"{s_fix:+13.3f} {np.linalg.norm(b2-a2):7.1f}")

print("""
Read this as: a genuine antipodal pair scores 2.0 on a max-2.0 scale.
If 'current best' never gets near +2 while 'sign-flipped' hits +2, the search
is currently maximising the wrong sign, and the widths it picks collapse to
roughly min_dist (a degenerate pair of nearby points on the same edge)
instead of spanning the object.
""")


# ----------------------------------------------------------------------
# CHECK 2: does plate_thickness=30 with min_dist=19 flip theta by 90 deg?
# ----------------------------------------------------------------------
print("=" * 74)
print("CHECK 2 — theta flip when predicted gripper width < plate_thickness")
print("=" * 74)
for w in [20, 25, 29, 31, 40, 60]:
    corners = xywh_theta_to_corners(300.0, 200.0, w, 30.0, 10.0)
    *_, theta_reported = convert(corners)
    flag = "  <-- 90 deg flip, angle check will fail" if abs(theta_reported - 10.0) > 45 else ""
    print(f"  gripper width {w:3d} px, plate 30 px, intended theta 10.0 deg "
          f"-> convert() reports {theta_reported:6.1f} deg{flag}")


# ----------------------------------------------------------------------
# CHECK 3: anisotropic resize — label frame vs. what the CNN actually sees
# ----------------------------------------------------------------------
print()
print("=" * 74)
print("CHECK 3 — 448x336 ROI squashed to 224x224 desyncs label from image")
print("=" * 74)
sx, sy = 224 / 448, 224 / 336
print(f"  scale_x = {sx:.3f}, scale_y = {sy:.3f}, anisotropy = {sy/sx:.3f}")


def which_edge(c):
    return 1 if np.linalg.norm(c[1] - c[0]) >= np.linalg.norm(c[2] - c[1]) else 2


# NOTE: replace these ranges with YOUR measured GT width/height distribution
W_RANGE, H_RANGE = (30, 80), (18, 40)
rng = np.random.default_rng(1)
flips, gaps_ok, gaps_flip = 0, [], []
N = 6000
for _ in range(N):
    w, h, th = rng.uniform(*W_RANGE), rng.uniform(*H_RANGE), rng.uniform(-90, 90)
    c = xywh_theta_to_corners(300, 240, w, h, th)
    c2 = c.copy()
    c2[:, 0] *= sx
    c2[:, 1] *= sy
    *_, t0 = convert(c)
    *_, t1 = convert(c2)
    d = abs(t0 - t1) % 180
    d = min(d, 180 - d)
    if which_edge(c) != which_edge(c2):
        flips += 1
        gaps_flip.append(d)
    else:
        gaps_ok.append(d)

print(f"  typical case ({100*len(gaps_ok)/N:.1f}% of rects): label theta and visible theta "
      f"differ by {np.mean(gaps_ok):.2f} deg on average, up to {np.max(gaps_ok):.2f} deg")
if gaps_flip:
    print(f"  contradictory case ({100*flips/N:.1f}% of rects): differ by "
          f"{np.mean(gaps_flip):.1f} deg on average (min {np.min(gaps_flip):.1f})")
print("""
  The angle budget in the correctness metric is 30 deg total. The first number
  is a systematic tax on every image; the second is supervision that actively
  points the wrong way. Both vanish if the ROI crop is square so that
  scale_x == scale_y.

  Re-run this with your real measured w/h ranges (W_RANGE / H_RANGE above)
  from the same script you used to pick min_dist/max_dist.
""")