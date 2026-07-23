#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
           points.append(np.nan, np.nan)


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


#convert rectangle to (x, y, w, h, theta)
def convert(rectangle):
   center = rectangle.mean(axis = 0)
   #vector
   edge = rectangle[1] - rectangle[0]


   w = np.linalg.norm(edge)
   h = np.linalg.norm(rectangle[2] - rectangle[1])
  
   theta = np.degrees(np.arctan2(edge[1], edge[0]))
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




  



