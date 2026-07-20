# imports
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt

# load the RGB photo of a given object ID
def load_rgb(base_path, pcd_id):
    # opened in BGR order
    img_bgr = cv2.imread(f"{base_path}/pcd{pcd_id}r.png")
    # BGR --> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

# load the depth image of a given object ID
def load_depth(base_path, pcd_id):
    # keep raw depth 
    depth = cv2.imread(f"{base_path}/pcd{pcd_id}d.tiff", cv2.IMREAD_UNCHANGED)
    return depth

# reads cpos.txt or cneg.txt file and coverts it into a list
def parse_grasp_rectangles(filepath):
    # open file and read every line
    with open(filepath, "r") as f: 
        lines = f.readlines()
    
    # convert each line into number pair
    points = []
    for line in lines: 
        parts = line.strip().split()

        # skip if not exactly 2 values
        if len(parts) != 2:
            continue

        try: 
            x, y = float(parts[0]), float(parts[1])
            points.append((x, y))
        except ValueError:
            points.append(np.nan, np.nan)

    # group 4 points into 1 rectangle 
    rectangles = []
    for i in range(0, len(points) - 3, 4):
        group = points[i : i + 4]

        # skip entirely 
        if any(np.isnan(p[0]) for p in group):
            continue

        # convert 4 tuples (x, y) into numpy array
        rectangles.append(np.array(group))

    return rectangles

# rectangle --> (x, y, w, h, theta)
def convert(rectangle):
    center = rectangle.mean(axis = 0)
    edge = rectangle[1] - rectangle[0]

    w = np.linalg.norm(edge)
    h = np.linalg.norm(rectangle[2] - rectangle[1])
    
    # direction the width-edge is pointing 
    theta = np.degrees(np.arctan2(edge[1], edge[0]))
    return center[0], center[1], w, h, theta

# loads one object's image and its positive grasp rectangles
def visualize(base_path, pcd_id):
    img = load_rgb(base_path, pcd_id)
    # only pos grasps
    pos_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")
    fig, ax = plt.subplots(1, figsize = (6, 6))
    ax.imshow(img)

    for rect in pos_rects: 
        closed = np.vstack([rect, rect[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth = 2)
    
    ax.set_title(f"pcd{pcd_id} — {len(pos_rects)} positive grasps")
    plt.show() 

if __name__ == "__main__":
    visualize("/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/01", "0105")


    
