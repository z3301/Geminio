"""Generate comparison figures for Geminio presentation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

QUERIES = [
    ("Any_jewelry", "\"Any jewelry\""),
    ("Any_human_faces", "\"Any human faces\""),
    ("Any_males_with_a_beard", "\"Any males with a beard\""),
    ("Any_guns", "\"Any guns\""),
    ("Any_females_riding_a_horse", "\"Any females riding a horse\""),
]

REC_LOSSES = {
    "baseline": 0.1828,
    "Any_jewelry": 0.0965,
    "Any_human_faces": 0.0750,
    "Any_males_with_a_beard": 0.1097,
    "Any_guns": 0.0984,
    "Any_females_riding_a_horse": 0.1222,
}

# --- Figure 1: Full overview (original + baseline + all 5 queries) ---
print("Generating full overview figure...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Geminio: Gradient Inversion Attack Results (128 Private ImageNet Samples)",
             fontsize=16, fontweight='bold', y=0.98)

panels = [
    ("original.jpg", "Ground Truth (128 Private Images)"),
    ("baseline.jpg", f"Baseline GIA (loss={REC_LOSSES['baseline']:.4f})"),
]
for key, label in QUERIES:
    panels.append((f"{key}.jpg", f"{label}\n(loss={REC_LOSSES[key]:.4f})"))

# hide last axis (2x4=8 slots, 7 panels)
axes[1, 3].axis('off')

for idx, (fname, title) in enumerate(panels):
    row, col = divmod(idx, 4)
    ax = axes[row][col]
    img = mpimg.imread(os.path.join(ASSETS, fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "full_overview.png"), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: {OUTDIR}/full_overview.png")

# --- Figure 2: Baseline vs Best Query (Human Faces) side-by-side ---
print("Generating baseline vs Geminio comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Baseline GIA vs. Geminio (Query: \"Any human faces\")",
             fontsize=14, fontweight='bold')

for ax, (fname, title) in zip(axes, [
    ("original.jpg", "Ground Truth"),
    ("baseline.jpg", f"Baseline (loss={REC_LOSSES['baseline']:.4f})"),
    ("Any_human_faces.jpg", f"Geminio: \"Any human faces\" (loss={REC_LOSSES['Any_human_faces']:.4f})"),
]):
    img = mpimg.imread(os.path.join(ASSETS, fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "baseline_vs_geminio.png"), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: {OUTDIR}/baseline_vs_geminio.png")

# --- Figure 3: All 5 query results in a row ---
print("Generating all queries figure...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.suptitle("Geminio Targeted Reconstructions Across 5 Semantic Queries",
             fontsize=14, fontweight='bold')

for ax, (key, label) in zip(axes, QUERIES):
    img = mpimg.imread(os.path.join(ASSETS, f"{key}.jpg"))
    ax.imshow(img)
    ax.set_title(f"{label}\n(loss={REC_LOSSES[key]:.4f})", fontsize=9)
    ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "all_queries.png"), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: {OUTDIR}/all_queries.png")

# --- Figure 4: Compact 2-row for Beamer slide (original+baseline top, queries bottom) ---
print("Generating compact Beamer figure...")
fig = plt.figure(figsize=(14, 7))

# Top row: original and baseline
ax1 = fig.add_subplot(2, 5, (1, 2))
ax1.imshow(mpimg.imread(os.path.join(ASSETS, "original.jpg")))
ax1.set_title("Ground Truth", fontsize=9, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(2, 5, (3, 4))
ax2.imshow(mpimg.imread(os.path.join(ASSETS, "baseline.jpg")))
ax2.set_title(f"Baseline GIA (loss={REC_LOSSES['baseline']:.4f})", fontsize=9, fontweight='bold')
ax2.axis('off')

# Architecture diagram in top-right
ax3 = fig.add_subplot(2, 5, 5)
ax3.imshow(mpimg.imread(os.path.join(ASSETS, "intro-git.png")))
ax3.set_title("Geminio Overview", fontsize=8)
ax3.axis('off')

# Bottom row: 5 queries
for i, (key, label) in enumerate(QUERIES):
    ax = fig.add_subplot(2, 5, 6 + i)
    ax.imshow(mpimg.imread(os.path.join(ASSETS, f"{key}.jpg")))
    ax.set_title(f"{label}\n(loss={REC_LOSSES[key]:.4f})", fontsize=7)
    ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "beamer_results.png"), dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: {OUTDIR}/beamer_results.png")

print("\nAll figures generated successfully!")
