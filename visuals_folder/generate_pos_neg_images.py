#imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data_loading import load_rgb, parse_grasp_rectangles, convert


#formats a single rectangle's (x, y, w, h, theta) into a readable label string
def format_rect_label(rect_corners, index=None):
    x, y, w, h, theta = convert(rect_corners)
    prefix = f"Rect {index}: " if index is not None else ""
    return f"{prefix}(x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, \u03B8={theta:.1f}\u00B0)"


#builds one figure: title on top, image sized to its TRUE aspect ratio (no
#letterboxing gaps), rectangles drawn in the given color, and every
#rectangle's (x,y,w,h,theta) listed directly below with no wasted space --
#figure height is computed in real inches from the actual content, instead
#of a guessed constant, which is what caused the big blank gap before
def make_annotated_figure(rgb, rects, color, pcd_id, save_path):
    img_h, img_w = rgb.shape[:2]
    aspect = img_w / img_h   # true width/height ratio of this specific photo

    fig_width_in = 6.0
    img_height_in = fig_width_in / aspect   # height that keeps the image undistorted

    title_height_in = 0.45
    line_height_in = 0.22
    text_block_height_in = line_height_in * len(rects) + 0.15

    fig_height_in = title_height_in + img_height_in + text_block_height_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    #place the axes at an EXACT position (in figure-fraction coordinates,
    #derived from the real inch measurements above) so the image fills
    #its box precisely -- this is what eliminates the letterboxing gap
    ax_bottom = text_block_height_in / fig_height_in
    ax_height = img_height_in / fig_height_in
    ax = fig.add_axes([0.0, ax_bottom, 1.0, ax_height])

    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for rect in rects:
        closed = np.vstack([rect, rect[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)

    #title, centered in the reserved space above the image
    title_y = ax_bottom + ax_height + (title_height_in * 0.5) / fig_height_in
    fig.text(0.5, title_y, f"pcd{pcd_id}", ha="center", va="center", fontsize=13, fontweight="bold")

    #rectangle labels, stacked tightly directly below the image
    for i, rect in enumerate(rects):
        label = format_rect_label(rect, index=i + 1)
        y_pos = (text_block_height_in - 0.12 - i * line_height_in) / fig_height_in
        fig.text(0.5, y_pos, label, ha="center", va="top", fontsize=9)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


#finds the first test-set image that has both positive AND negative
#rectangles, then generates both annotated images from it
def generate_pos_neg_images(split_path="dataset_split.json",
                             pos_output="positive_rectangle_example.png",
                             neg_output="negative_rectangles_example.png"):
    with open(split_path, "r") as f:
        split = json.load(f)

    for entry in split["test"]:
        pcd_id, folder = entry["id"], entry["folder"]

        pos_path = f"{folder}/pcd{pcd_id}cpos.txt"
        neg_path = f"{folder}/pcd{pcd_id}cneg.txt"

        if not os.path.exists(pos_path) or not os.path.exists(neg_path):
            continue

        pos_rects = parse_grasp_rectangles(pos_path)
        neg_rects = parse_grasp_rectangles(neg_path)

        if len(pos_rects) == 0 or len(neg_rects) == 0:
            continue

        rgb = load_rgb(folder, pcd_id)

        print(f"Using pcd{pcd_id} ({len(pos_rects)} positive, {len(neg_rects)} negative rectangles)")
        make_annotated_figure(rgb, pos_rects, "green", pcd_id, pos_output)
        make_annotated_figure(rgb, neg_rects, "red", pcd_id, neg_output)
        return pcd_id

    print("No test-set image found with both positive and negative rectangles.")
    return None


if __name__ == "__main__":
    generate_pos_neg_images()