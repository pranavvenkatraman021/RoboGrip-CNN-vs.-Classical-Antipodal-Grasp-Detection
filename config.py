import os

#dataset path
DATA_ROOT = os.environ.get(
    "CORNELL_ROOT",
    "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data",
)

BACKGROUNDS_DIR = os.path.join(DATA_ROOT, "backgrounds")

#output of baseline_sweep.py (tuned params)
MIN_DIST = 20
MAX_DIST = 90
PLATE_THICKNESS = 26

#ratio to increase size of rectangle 
WIDTH_MARGIN = 1.8 #1.15

#type of tie breaking with 0 tolerance 
TIE_BREAK = "first" # centroid
TIE_TOL = 0.0

#whether to reject pairs based on midpoint location
MASK_CHECK = False # false

#convert() measures theta from longer side, so cap to remove 90 degree flip
PLATE_CAP_RATIO = 0.95
