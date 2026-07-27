#imports
import matplotlib.pyplot as plt

#the accuracy progression across each fix, in order
labels = ["Pre-Fixes", "+BG Subtraction", "+ROI Fix", "+Angle Fix",
          "+Param Wiring", "+Sign Fix", "+Retuned"]
values = [14.0, 13.53, 17.05, 24.81, 27.91, 28.80, 32.03]

#colors group the stages of baseline development
colors = [
    "tab:red",                                      #pre-fixes
    "tab:orange", "tab:orange", "tab:orange",       #BG through angle fix
    "tab:green", "tab:green", "tab:green"           #remaining fixes
]

fig, ax = plt.subplots()

bars = ax.bar(labels, values, color=colors)

#value labels directly above each bar
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val:.2f}%",
        ha="center",
        va="bottom"
    )

ax.set_ylabel("Accuracy (%)")
ax.set_title("Baseline Accuracy Across Fixes")

#rotate bottom labels to prevent overlap
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("baseline_accuracy_progression.png", dpi=200)
plt.close(fig)

print("Saved: baseline_accuracy_progression.png")