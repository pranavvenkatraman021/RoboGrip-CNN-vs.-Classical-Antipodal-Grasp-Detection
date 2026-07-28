#imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#your actual measured results -- update these if you re-run evaluate_by_category.py
groups    = ["Simple\nConvex", "Elongated\n& Tools", "Handled\n& Irregular",
             "Flat &\nDeformable", "Complex\n& Misc"]
baseline  = [31.0, 18.9, 36.8, 51.5,  6.2]
cnn       = [62.1, 75.7, 84.2, 90.9, 62.5]
ns        = [29,   37,   19,   33,   16  ]

#why the gap differs -- one clause per group, written to read naturally out loud
why = [
    "Geometry is simple;\nboth methods struggle",
    "Asymmetric shapes;\nno clean antipodal pair",
    "Handles break the\ngeometric rule",
    "Flat surfaces help\nthe baseline most",
    "Most complex objects;\nbaseline near zero",
]

NAVY  = "#1D3461"
GREEN = "#1B7F4F"
GAP   = "#C0392B"
GREY  = "#ECF0F1"
TEXT  = "#2C3E50"
LIGHT = "#FAFBFC"

fig = plt.figure(figsize=(13, 7.8))
fig.patch.set_facecolor(LIGHT)

#main chart takes up the top 78% of the figure
ax = fig.add_axes([0.07, 0.22, 0.90, 0.70])
ax.set_facecolor(LIGHT)

x     = np.arange(len(groups))
width = 0.30

#bars -- slightly desaturated so the gap annotations pop
bars_b = ax.bar(x - width/2 - 0.02, baseline, width,
                color=NAVY, alpha=0.88, zorder=3)
bars_c = ax.bar(x + width/2 + 0.02, cnn,      width,
                color=GREEN, alpha=0.88, zorder=3)

#subtle value labels -- small, inside the top of each bar
for bar, val in zip(bars_b, baseline):
    if val > 8:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 3.5,
                f"{val:.0f}%", ha="center", va="top",
                fontsize=8.5, color="white", fontweight="bold")
for bar, val in zip(bars_c, cnn):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 3.5,
            f"{val:.0f}%", ha="center", va="top",
            fontsize=8.5, color="white", fontweight="bold")

#gap line connecting each pair -- the main visual argument
for i, (b, c) in enumerate(zip(baseline, cnn)):
    gap = c - b
    #thin vertical line connecting the top of each bar
    bx = x[i] - width/2 - 0.02 + width/2
    cx = x[i] + width/2 + 0.02 + width/2
    ax.plot([bx, cx], [b, c], color=GAP, lw=1.5,
            alpha=0.6, zorder=4, ls="--")
    #gap badge above the midpoint
    mx = (bx + cx) / 2
    my = max(b, c) + 5.5
    ax.text(mx, my, f"+{gap:.0f}pp",
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=GAP)

#n labels below bars, very quiet
for i, n in enumerate(ns):
    ax.text(i, -2.8, f"n={n}", ha="center", va="top",
            fontsize=8, color="#95A5A6")

#thin horizontal reference lines, labelled inline
for val, col, label in [
    (32.03, NAVY,  "Baseline: 32%"),
    (72.39, GREEN, "CNN: 72%"),
]:
    ax.axhline(val, color=col, lw=0.9, ls=":", alpha=0.45, zorder=1)
    ax.text(4.47, val + 1.2, label, ha="right", va="bottom",
            fontsize=8, color=col, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11.5, color=TEXT)
ax.set_ylim(-6, 108)
ax.set_ylabel("Accuracy (%)", fontsize=11.5, color=TEXT, labelpad=8)
ax.set_title("Where Does Learning Help?",
             fontsize=16, fontweight="bold", color=TEXT, pad=14)

ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(axis="both", color="#CCCCCC", labelcolor=TEXT)
ax.yaxis.set_tick_params(length=0)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", color="#E0E0E0", lw=0.7, zorder=0)
ax.set_axisbelow(True)

#legend -- minimal, top left
from matplotlib.patches import Patch
handles = [Patch(facecolor=NAVY,  alpha=0.88, label="Classical baseline"),
           Patch(facecolor=GREEN, alpha=0.88, label="CNN (multi-GT)")]
ax.legend(handles=handles, loc="upper left", fontsize=10,
          frameon=False, handlelength=1.2, handleheight=0.85)

#"why" strip -- plain text boxes along the bottom, same width as each group
for i, (txt, b, c) in enumerate(zip(why, baseline, cnn)):
    gap = c - b
    #shade very lightly based on gap size so bigger gaps stand out softly
    alpha = 0.10 + 0.22 * (gap / 60)
    fig.add_axes([0.07 + i * 0.183, 0.015, 0.17, 0.18],
                 facecolor=GREY).set_visible(False)

    #use figure text positioned to align under each group
    fx = 0.07 + 0.04 + i * 0.183 + 0.045
    fig.text(fx, 0.11, txt,
             ha="center", va="center", fontsize=8.2,
             color="#555555", multialignment="center", linespacing=1.5,
             transform=fig.transFigure)
    fig.text(fx, 0.185, f"+{c-b:.0f}pp gap",
             ha="center", va="center", fontsize=8,
             color=GAP, fontweight="bold",
             transform=fig.transFigure)

#divider line above the why strip
fig.add_artist(plt.Line2D([0.07, 0.97], [0.205, 0.205],
               color="#DDDDDD", lw=0.8, transform=fig.transFigure))

#"Why:" label on the left of the strip
fig.text(0.015, 0.11, "Why\neach\ngap:",
         ha="center", va="center", fontsize=8, color="#888",
         fontweight="bold", multialignment="center")

plt.savefig("category_breakdown_clean.png",
            dpi=200, bbox_inches="tight", facecolor=LIGHT)
plt.close()
print("saved")