"""Vẽ hình cho report.md từ các CSV đã lưu. Không huấn luyện lại, không chấm lại.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/d_figures.py
Đầu ra: reports/figures/report_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from common import *  # noqa: F403

FIG = NB_DIR / "reports" / "figures"
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
NB_COL, SVM_COL = "#2a78d6", "#eb6834"                 # đã qua validate_palette.js --pairs all
RAMP = LinearSegmentedColormap.from_list("blue", ["#eaf2fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": HAIR,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0})


def confusion_panel(ax, cm, title, show_y=True):
    cm = np.asarray(cm)
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap=RAMP, vmin=0, vmax=1, aspect="equal")
    for i in range(C):
        for j in range(C):
            col = "white" if frac[i, j] >= 0.65 else INK
            ax.text(j, i - 0.09, str(cm[i, j]), ha="center", va="center", color=col, fontsize=12, fontweight="semibold")
            ax.text(j, i + 0.21, f"{frac[i, j]:.0%}", ha="center", va="center", color=col, fontsize=8.5)
    ax.set_xticks(range(C), LABELS)
    ax.set_yticks(range(C), LABELS if show_y else [""] * C)
    ax.set_xticks(np.arange(-0.5, C), minor=True)
    ax.set_yticks(np.arange(-0.5, C), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Nhãn dự đoán")
    if show_y:
        ax.set_ylabel("Nhãn thật")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)


def read_cm(rel):
    return pd.read_csv(NB_DIR / rel, index_col=0).loc[LABELS, LABELS].to_numpy()


# Hình 1: ma trận nhầm lẫn của NB trên test 287 mẫu
panels = [("experiments_report/b_confusion_BoW_val_fit_trainval.csv", "BoW + NB"),
          ("experiments_report/b_confusion_TFIDF_val_fit_trainval.csv", "TF-IDF + NB (R0)"),
          ("experiments_v2/final_film_v2_confusion_NB_v2.csv", "NB cuối (V2)")]
fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
for i, (ax, (p, t)) in enumerate(zip(axes, panels)):
    confusion_panel(ax, read_cm(p), t, show_y=i == 0)
fig.savefig(FIG / "report_nb_confusion.png", dpi=200)

# Hình 2: NB và SVM trên cùng split, cùng đặc trưng
fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharex=True, sharey=True, constrained_layout=True)
for ax, tag, title in zip(axes, ("splitD", "splitH"), ("Split D: test 418 dòng", "Split H: test 287 dòng")):
    r = pd.read_csv(OUT / f"c_{tag}_results.csv").set_index("model")
    for y0, feat in ((1, "BoW"), (0, "TF-IDF")):
        for off, algo, col in ((0.13, "NB", NB_COL), (-0.13, "SVM", SVM_COL)):
            m = r.loc[f"{algo} {feat}"]
            ax.plot([m.ci95_lo, m.ci95_hi], [y0 + off] * 2, color=col, lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot(m.macro_f1, y0 + off, "o", color=col, ms=8, mec=SURFACE, mew=1.2, zorder=3)
    ax.set_yticks([1, 0], ["BoW", "TF-IDF"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0.44, 0.72)
    ax.grid(axis="x", color=HAIR, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Macro-F1 (chấm tròn), CI95 bootstrap theo nhóm (đoạn thẳng)")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
fig.legend(handles=[Line2D([], [], color=c, marker="o", ms=8, lw=1.8, mec=SURFACE, label=l) for c, l in ((NB_COL, "Naive Bayes"), (SVM_COL, "SVM"))],
           loc="outside upper right", ncol=2, frameon=False)
fig.savefig(FIG / "report_nb_vs_svm_same_split.png", dpi=200)

# Hình 3: ROC theo từng lớp, split D
pts = pd.read_csv(OUT / "c_splitD_roc_points.csv.gz")
res = pd.read_csv(OUT / "c_splitD_results.csv").set_index("model")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharex=True, sharey=True, constrained_layout=True)
for row, feat in enumerate(("BoW", "TF-IDF")):
    for col, lab in enumerate(LABELS):
        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], color="#bdbcb6", lw=0.8, zorder=1)
        for algo, c_ in (("NB", NB_COL), ("SVM", SVM_COL)):
            d = pts[(pts.model == f"{algo} {feat}") & (pts.label == lab)]
            ax.plot(d.fpr, d.tpr, color=c_, lw=1.7, drawstyle="default", zorder=2)
        a, b = res.loc[f"NB {feat}", f"auc_{lab}"], res.loc[f"SVM {feat}", f"auc_{lab}"]
        ax.text(0.97, 0.05, f"AUC NB {a:.3f}\nAUC SVM {b:.3f}", ha="right", va="bottom", transform=ax.transAxes, color=INK2, fontsize=9, linespacing=1.4)
        ax.grid(color=HAIR, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{feat}, lớp {lab}", fontsize=10.5, color=INK, loc="left")
        if col == 0:
            ax.set_ylabel("True positive rate")
        if row == 1:
            ax.set_xlabel("False positive rate")
fig.legend(handles=[Line2D([], [], color=c, lw=1.8, label=l) for c, l in ((NB_COL, "Naive Bayes"), (SVM_COL, "SVM"))] + [Line2D([], [], color="#bdbcb6", lw=0.8, label="Đoán ngẫu nhiên")],
           loc="outside upper right", ncol=3, frameon=False)
fig.savefig(FIG / "report_roc_nb_svm.png", dpi=200)

# Hình 4: ma trận nhầm lẫn NB và SVM, TF-IDF, split D
fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), constrained_layout=True)
confusion_panel(axes[0], read_cm("experiments_report/c_splitD_confusion_NB TF-IDF.csv"), "TF-IDF + NB, split D")
confusion_panel(axes[1], read_cm("experiments_report/c_splitD_confusion_SVM TF-IDF.csv"), "TF-IDF + SVM, split D", show_y=False)
fig.savefig(FIG / "report_confusion_nb_svm.png", dpi=200)
print("Đã lưu 4 hình vào", FIG)
