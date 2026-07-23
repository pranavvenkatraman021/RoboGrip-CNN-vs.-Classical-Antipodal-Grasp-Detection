#imports
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_loading import load_depth, load_rgb, parse_grasp_rectangles


#loads every background photo once
def load_backgrounds(backgrounds_dir):
   bg_paths = glob.glob(os.path.join(backgrounds_dir, "*r.png"))
   backgrounds = []
   for path in bg_paths:
       #convert to RGB
       bg_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
       backgrounds.append(bg_img)
   return backgrounds




#picks whichever background photo best matches this image's outer border
def find_best_matching_background(img, backgrounds, border=20):
   h, w = img.shape[:2]


   #mark just the border strip as True, everything else False
   border_mask = np.zeros((h, w), dtype=bool)
   border_mask[:border, :] = True    # top strip
   border_mask[-border:, :] = True   # bottom strip
   border_mask[:, :border] = True    # left strip
   border_mask[:, -border:] = True   # right strip


   best_score = np.inf
   best_bg = None


   for bg in backgrounds:
       #skip if sizes don't match
       if bg.shape != img.shape:
           continue 


       #average color difference, but only within the border region
       diff = np.abs(img.astype(int) - bg.astype(int))
       border_diff = diff[border_mask].mean()


       if border_diff < best_score:
           best_score = border_diff
           best_bg = bg


   return best_bg


#RGB image into a mask (white is object, black is background)
def create_mask(img, backgrounds):


   #find the background photo that matches this scene
   best_bg = find_best_matching_background(img, backgrounds)
   if best_bg is None:
       raise ValueError("No matching background found — check image sizes match.")


   #per-pixel color difference, summed across R,G,B into one grayscale map
   diff = np.abs(img.astype(int) - best_bg.astype(int)).sum(axis=2)
   diff = diff.astype(np.uint8)


   #otsu's thresholding
   _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


   #clean up small noisy specks
   kernel = np.ones((5, 5), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


   return mask


#find contour of object in the mask
def get_largest_contour(mask):
   #trace outlines of white regions in mask
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


   if len(contours) == 0:
       return None
  
   #pick largest contour area
   largest = max(contours, key = cv2.contourArea)
   #(N, 2) array of (x, y) points
   return largest.squeeze()


#estimates surface normal
def estimate_normals(contour, step = 8):
   #step is how many points ahead/behind
   n_points = len(contour)
   normals = []


   for i in range(n_points):
       #points a bit before and after
       p_before = contour[(i - step) % n_points]
       p_after = contour[(i + step) % n_points]


       tangent = p_after.astype(float) - p_before.astype(float)
       tangent_len = np.linalg.norm(tangent)


       if tangent_len == 0:
           normals.append(np.array([0.0, -1.0]))
           continue


       tangent = tangent / tangent_len
       #normal is tangent rotated by 90 degrees
       normal = np.array([-tangent[1], tangent[0]])
       normals.append(normal)
  
   return normals


#searches pairs of contour points to find best antipodal pair
def find_best_antipodal_pair(contour, normals, min_dist, max_dist, sample_step = 4):
   n_points = len(contour)
   best_score = -np.inf
   best_pair = None


   #only look at subset
   indices = range(0, n_points, sample_step)


   for i in indices:
       for j in indices:
           if i >= j:
               continue


           point_a = contour[i].astype(float)
           point_b = contour[j].astype(float)


           dist = np.linalg.norm(point_b - point_a)


           #skip anything too close or too far (real gripper)
           if dist < min_dist or dist > max_dist:
               continue


           direction_ab = (point_b - point_a) / dist


           normal_a = normals[i]
           normal_b = normals[j]


           #dot product of normals with two scores (one for each)
           score_a = np.dot(normal_a, direction_ab)
           score_b = np.dot(normal_b, -direction_ab)


           #combine both scores
           score = score_a + score_b


           if score > best_score:
               best_score = score
               best_pair = (point_a, point_b)


   return best_pair, best_score


#convert from antipodal lines to a full rectangle
def pair_to_grasp_rectangle(point_a, point_b, plate_thickness = 25):
   center = (point_a + point_b) / 2
   w = np.linalg.norm(point_b - point_a)
   h = plate_thickness
  
   edge = point_b - point_a
   theta = np.degrees(np.arctan2(edge[1], edge[0]))


   return center[0], center[1], w, h, theta


#converts (x, y, w, h) to 4 rectangle
def xywh_theta_to_corners(x, y, w, h, theta_deg):
   theta = np.radians(theta_deg)
   dx_w, dy_w = np.cos(theta) * (w / 2), np.sin(theta) * (w / 2)
   dx_h, dy_h = -np.sin(theta) * (h / 2), np.cos(theta) * (h / 2)


   corners = np.array([
       [x - dx_w - dx_h, y - dy_w - dy_h],
       [x + dx_w - dx_h, y + dy_w - dy_h],
       [x + dx_w + dx_h, y + dy_w + dy_h],
       [x - dx_w + dx_h, y - dy_w + dy_h],
   ])


   return corners


#runnining full baseline on one object + visualization
def run_baseline(base_path, pcd_id, backgrounds_dir, min_dist = 30, max_dist = 150):
   img = load_rgb(base_path, pcd_id)
   #depth is no longer needed for masking — RGB background subtraction replaced it


   backgrounds = load_backgrounds(backgrounds_dir)
   mask = create_mask(img, backgrounds)  
   contour = get_largest_contour(mask)


   if contour is None or len(contour) < 10:
       print(f"pcd{pcd_id}: couldn't find a usable contour — skipping.")
       return


   normals = estimate_normals(contour)
   best_pair, score = find_best_antipodal_pair(contour, normals, min_dist, max_dist)


   if best_pair is None:
       print(f"pcd{pcd_id}: no valid antipodal pair found — try adjusting min_dist/max_dist.")
       return


   point_a, point_b = best_pair
   x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b)
   predicted_corners = xywh_theta_to_corners(x, y, w, h, theta)


   #load ground truth rectangles to visually compare
   gt_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")


   fig, ax = plt.subplots(1, figsize=(6, 6))
   ax.imshow(img)


   for rect in gt_rects:
       closed = np.vstack([rect, rect[0]])
       ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth=1.5, label="ground truth" if rect is gt_rects[0] else None)


   closed_pred = np.vstack([predicted_corners, predicted_corners[0]])
   ax.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2, label="baseline prediction")


   ax.set_title(f"pcd{pcd_id} — baseline antipodal score: {score:.2f}")
   ax.legend()
   plt.show()


if __name__ == "__main__":
   run_baseline(
       "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/02",
       "0202",
       "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/backgrounds"
   )











