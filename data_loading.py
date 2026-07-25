#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

#defines a centered rectangular region, excluding a margin from each edge --
#mirrors the same ROI insight from baseline.py's classical pipeline
#(see get_roi_bounds there): objects in this dataset are always placed
#centrally, so cropping out the margin before resizing gives the CNN much
#more effective resolution on the object, instead of shrinking mostly
#board/background down to 224x224 along with it.
#Duplicated here (not imported from baseline.py) specifically to avoid
#a circular import, since baseline.py already imports from this file.
def get_roi_bounds(img_shape, margin_frac=0.15):
    h, w = img_shape[:2]
    x_margin = int(w * margin_frac)
    y_margin = int(h * margin_frac)
    return x_margin, w - x_margin, y_margin, h - y_margin

#crops an image down to the central ROI, returns the crop plus the
#(x_offset, y_offset) needed to map crop-local coordinates back to
#the original full-image coordinates later
def crop_to_roi(img, margin_frac=0.15):
    x_min, x_max, y_min, y_max = get_roi_bounds(img.shape, margin_frac)
    return img[y_min:y_max, x_min:x_max], x_min, y_min

#loads RGB photo for given object ID
def load_rgb(base_path, pcd_id):
   #loads in BGR
   img_bgr = cv2.imread(f"{base_path}/pcd{pcd_id}r.png")
   #convert to RGB
   img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
   return img_rgb

#loads depth image for given object ID
def load_depth(base_path, pcd_id):
   #keep raw depth values
   depth = cv2.imread(f"{base_path}/pcd{pcd_id}d.tiff", cv2.IMREAD_UNCHANGED)
   return depth

#reads cpos or cneg anc convets it to list of rectangles
def parse_grasp_rectangles(filepath):
   #open file and read to list of strings
   with open(filepath, "r") as f:
       lines = f.readlines()
  
   #convert each line to (x, y)
   points = []
   for line in lines:
       parts = line.strip().split()

       #check for lines with missing values
       if len(parts) != 2:
           continue
       try:
           x, y = float(parts[0]), float(parts[1])
           points.append((x, y))
       except ValueError:
           points.append((np.nan, np.nan))

   #every four points becomes a rectangle
   rectangles = []
   for i in range(0, len(points) - 3, 4):
       group = points[i : i + 4]

       #skip any rectangle with NaN
       if any(np.isnan(p[0]) for p in group):
           continue

       #four tuples becomes numpy array (rectangle)
       rectangles.append(np.array(group))

   return rectangles
"""
#convert rectangle to (x, y, w, h, theta)
def convert(rectangle):
   center = rectangle.mean(axis = 0)
   #vector
   edge = rectangle[1] - rectangle[0]

   w = np.linalg.norm(edge)
   h = np.linalg.norm(rectangle[2] - rectangle[1])
  
   theta = np.degrees(np.arctan2(edge[1], edge[0]))
   return center[0], center[1], w, h, theta

   """

def convert(corners):
    center = corners.mean(axis=0)

    edge1 = corners[1] - corners[0]
    edge2 = corners[2] - corners[1]

    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)

    if len1 >= len2:
        w, h, main_edge = len1, len2, edge1
    else:
        w, h, main_edge = len2, len1, edge2

    theta = np.degrees(np.arctan2(main_edge[1], main_edge[0]))
    return center[0], center[1], w, h, theta

#loads image and positive grasp rectangles
def visualize(base_path, pcd_id):
   img = load_rgb(base_path, pcd_id)
   pos_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")
   fig, ax = plt.subplots(1, figsize = (6, 6))
   ax.imshow(img)

   #draw each rectangle
   for rect in pos_rects:
       closed = np.vstack([rect, rect[0]])
       ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth = 2)
  
   ax.set_title(f"pcd{pcd_id} — {len(pos_rects)} positive grasps")
   plt.show()

if __name__ == "__main__":
   visualize("/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/01", "0105")




  



