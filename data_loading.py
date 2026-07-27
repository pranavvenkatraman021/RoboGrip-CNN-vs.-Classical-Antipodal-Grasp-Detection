#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt




#replica on the one from baseline.py to avoid circular import 
def get_roi_bounds(img_shape, margin_frac=0.15):
    h, w = img_shape[:2]
    x_margin = int(w * margin_frac)
    y_margin = int(h * margin_frac)
    return x_margin, w - x_margin, y_margin, h - y_margin


#crops an image down to the central ROI
def crop_to_roi(img, margin_frac=0.15):
    x_min, x_max, y_min, y_max = get_roi_bounds(img.shape, margin_frac)
    return img[y_min:y_max, x_min:x_max], x_min, y_min


#resizes without stretching and pads to a square
def letterbox_to_square(img, output_size=224):
    h, w = img.shape[:2]
    scale = min(output_size / w, output_size / h)


    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))


    pad_x = (output_size - new_w) // 2
    pad_y = (output_size - new_h) // 2
    pad_right = output_size - new_w - pad_x
    pad_bottom = output_size - new_h - pad_y


    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_bottom,
        pad_x,
        pad_right,
        cv2.BORDER_REFLECT_101
    )


    return padded, scale, pad_x, pad_y


#moves rectangle points into the padded square
def points_to_letterbox(points, scale, pad_x, pad_y):
    transformed = points.astype(np.float32).copy()
    transformed[:, 0] = transformed[:, 0] * scale + pad_x
    transformed[:, 1] = transformed[:, 1] * scale + pad_y
    return transformed


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


#converts corners into (x, y, w, h, theta)
#shared between the baseline and CNN
def convert(corners):
    center = corners.mean(axis=0)


    edge1 = corners[1] - corners[0]
    edge2 = corners[2] - corners[1]


    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)


    #assign w to longer edge and h to shorter one 
    if len1 >= len2:
        w, h, main_edge = len1, len2, edge1
    else:
        w, h, main_edge = len2, len1, edge2


    #gripper direction
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

