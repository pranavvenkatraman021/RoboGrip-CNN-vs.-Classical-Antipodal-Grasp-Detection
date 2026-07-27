#imports
import matplotlib.pyplot as plt

#the accuracy progression across each fix, in order
labels = ["Pre-Fixes", "+BG Subtraction", "+ROI Fix", "+Angle Fix"]
values = [14.0, 13.53, 17.05, 24.81]

fig, ax = plt.subplots()

#pure matplotlib defaults -- no custom colors, no custom fonts, no custom styling
bars = ax.bar(labels, values)

#value labels directly above each bar
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(),
        f"{val:.2f}%", ha="center", va="bottom"
    )

ax.set_ylabel("Accuracy (%)")
ax.set_title("Baseline Accuracy Across Fixes")

plt.tight_layout()
plt.savefig("baseline_accuracy_progression.png", dpi=200)
plt.close(fig)
print("Saved: baseline_accuracy_progression.png")