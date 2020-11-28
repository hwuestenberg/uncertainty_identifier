import numpy as np
import matplotlib.pyplot as plt
from uncert_ident.utilities import TRUE_NEGATIVE, TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE
from uncert_ident.visualisation.plotter import confusion_cmap, empty_plot, latex_textwidth, cblack, cgrey, cwhite, save



# true_classes = [r"\textsf{P}", r"\textsf{N}"]
# pred_classes = [r"\textsf{P}", r"\textsf{N}"]
true_classes = [r"$y=1$", r"$y=0$"]
pred_classes = [r"$y=1$", r"$y=0$"]
confusion = np.array([[TRUE_POSITIVE, FALSE_NEGATIVE],
                      [FALSE_POSITIVE, TRUE_NEGATIVE]])
confusion_str = np.array([[r"\textsf{TP}", r"\textsf{FN}"],
                          [r"\textsf{FP}", r"\textsf{TN}"]])


fig, ax = empty_plot(figwidth=latex_textwidth*0.6)
im = ax.imshow(confusion, cmap=confusion_cmap)

# We want to show all ticks...
ax.set_xticks(np.arange(len(true_classes)))
ax.set_yticks(np.arange(len(pred_classes)))
# ... and label them with the respective list entries
ax.set_xticklabels(true_classes)
ax.set_yticklabels(pred_classes)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(true_classes)):
    for j in range(len(pred_classes)):
        text = ax.text(j, i, confusion_str[i, j],
                       ha="center", va="center", color=cblack)


ax.set_xlabel(r"\textsf{Model prediction}")
ax.set_ylabel(r"\textsf{Ground truth}")
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
ax.xaxis.set_label_position("top")

# ax.set_title("Harvest of local farmers (in tons/year)")
ax.set_aspect("equal")
save("../figures/confusion_reference_beamer.pdf")
plt.show()
