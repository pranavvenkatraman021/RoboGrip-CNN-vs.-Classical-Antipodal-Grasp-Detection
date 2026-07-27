"""
Single place for the dataset path and the tuned baseline parameters.

Set CORNELL_ROOT once per session instead of editing four files:

    export CORNELL_ROOT=/content/cornell          # Colab / Linux
    export CORNELL_ROOT="/Users/pranavvenkatraman/Downloads/Cornell Grasp Data"

or in a Colab cell:

    import os
    os.environ["CORNELL_ROOT"] = "/content/cornell"
"""
import os

DATA_ROOT = os.environ.get(
    "CORNELL_ROOT",
    "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data",
)

BACKGROUNDS_DIR = os.path.join(DATA_ROOT, "backgrounds")


# --- tuned baseline parameters -------------------------------------------
# These are the ONLY place these values live. baseline.py and
# evaluate_baseline.py both read them, so a tuning decision cannot
# silently fail to reach the code that produces the final number.
#
# Replace these with whatever baseline_sweep.py picks.
MIN_DIST = 20
MAX_DIST = 90
PLATE_THICKNESS = 26

# --- rectangle shape ---
# Predicted rectangles measured ~15% smaller than ground truth in BOTH
# dimensions (width 49.3 vs 57.5 px, plate 26 vs 30.7 px on val). The chord
# between the two contact points is the OBJECT's width; a labelled gripper
# opening is wider than the object it closes on. Set from the measured ratio,
# then confirm against measure_gt_stats.py -- not tuned upward for its own sake.
WIDTH_MARGIN = 1.8 #1.15

# --- tie-breaking (finding B4) ---
# With the score sign fixed, near-perfect scores are common: every pair of
# points on two parallel edges scores ~2.0. "first" keeps whichever contour
# traversal reached first, i.e. an arbitrary position along the object.
#   "first" | "centroid" | "narrow" | "wide"
TIE_BREAK = "first" # centroid
TIE_TOL = 0.0

# Reject candidate pairs whose midpoint falls off the object. Helps on concave
# shapes (a chord across a mug can cross the handle hole) but can eliminate
# every candidate on some images, turning a prediction into a skip. Swept by
# baseline_sweep.py against the fixed denominator -- set from what it picks.
MASK_CHECK = False # false

# --- how to prevent the 90-degree theta flip (finding B2) ---
# convert() measures theta from the LONGER edge, so if the plate is longer than
# the gripper opening the reported angle rotates 90 degrees. Two ways to make
# that impossible:
#
#   PLATE_CAP_RATIO = 0.95  cap the plate at 0.95x the opening. Narrow objects
#                           still get a prediction, just with a shorter plate.
#                           (pcd0253's widest antipodal chord is 27 px -- under
#                           the clamp approach it produced nothing at all.)
#   PLATE_CAP_RATIO = None  fall back to clamping min_dist up to plate_thickness,
#                           which forbids grasps narrower than the plate entirely.
#
# NOTE: this is NOT the "scale plate_thickness proportionally with width" idea
# that was tested and rejected earlier -- that rescaled EVERY rectangle. This
# only binds on the narrow tail, leaving normal-width grasps untouched.
PLATE_CAP_RATIO = 0.95

# Hard requirement (finding B2): if MIN_DIST < PLATE_THICKNESS, any predicted
# rectangle narrower than the plate has its longer edge running along the
# plate instead of the closing direction, so convert() reports a theta
# rotated by 90 degrees and the angle check fails automatically.