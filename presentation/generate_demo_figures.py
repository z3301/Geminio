"""Generate demonstration figures for all 3 domains (ImageNet, Medical, UAV).

Produces composite comparison images similar to the original Geminio README:
  - Ground truth vs baseline vs query-targeted reconstructions
  - Per-domain and cross-domain comparison panels

Usage:
    python presentation/generate_demo_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, 'assets')
RESULTS = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(ROOT, 'presentation', 'figures')
os.makedirs(OUTDIR, exist_ok=True)


def load_metrics(result_dir):
    """Load metrics.json from a result directory."""
    path = os.path.join(RESULTS, result_dir, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def add_metrics_text(ax, metrics, keys=('lpips', 'psnr')):
    """Add metrics as text annotation below the image."""
    parts = []
    if 'lpips' in keys and 'lpips' in metrics:
        parts.append(f"LPIPS={metrics['lpips']:.3f}")
    if 'psnr' in keys and 'psnr' in metrics:
        parts.append(f"PSNR={metrics['psnr']:.1f}")
    if 'attack_f1' in keys and 'attack_f1' in metrics:
        parts.append(f"F1={metrics['attack_f1']:.3f}")
    if parts:
        ax.text(0.5, -0.02, '  |  '.join(parts), transform=ax.transAxes,
                ha='center', va='top', fontsize=7, color='#555555',
                fontfamily='monospace')


# =============================================================================
# Figure 1: ImageNet Domain (original Geminio reproduction)
# =============================================================================
print("=== Figure 1: ImageNet Domain ===")

IMAGENET_QUERIES = [
    ("Any_jewelry", '"Any jewelry"'),
    ("Any_human_faces", '"Any human faces"'),
    ("Any_males_with_a_beard", '"Males with a beard"'),
    ("Any_guns", '"Any guns"'),
    ("Any_females_riding_a_horse", '"Females riding a horse"'),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 16))
fig.suptitle("ImageNet Domain: Geminio Attack Reproduction\n"
             "128 Private Images | ResNet34 | CLIP ViT-L/14",
             fontsize=14, fontweight='bold', y=0.98)

# Row 1: Ground truth
img = mpimg.imread(os.path.join(ASSETS, 'original.jpg'))
axes[0].imshow(img)
axes[0].set_title("Ground Truth (128 Private ImageNet Images)", fontsize=11, pad=8)
axes[0].axis('off')

# Row 2: Baseline
img = mpimg.imread(os.path.join(ASSETS, 'baseline.jpg'))
axes[1].imshow(img)
axes[1].set_title("Baseline Gradient Inversion (No Targeting)", fontsize=11, pad=8)
axes[1].axis('off')

# Row 3: Best query (human faces)
img = mpimg.imread(os.path.join(ASSETS, 'Any_human_faces.jpg'))
axes[2].imshow(img)
axes[2].set_title('Geminio: "Any human faces" (Targeted Reconstruction)', fontsize=11, pad=8)
axes[2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, 'demo_imagenet.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: demo_imagenet.png")

# Also: all 5 queries in a grid
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("ImageNet: All 5 Targeted Queries", fontsize=13, fontweight='bold')
for ax, (key, label) in zip(axes, IMAGENET_QUERIES):
    img = mpimg.imread(os.path.join(ASSETS, f'{key}.jpg'))
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'demo_imagenet_queries.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  Saved: demo_imagenet_queries.png")

# =============================================================================
# Figure 2: Medical Domain (ChestMNIST + BiomedCLIP)
# =============================================================================
print("\n=== Figure 2: Medical Domain ===")

med_baseline_m = load_metrics('medical_baseline')
med_pneumonia_m = load_metrics('medical_pneumonia_descriptive')
med_effusion_m = load_metrics('medical_effusion_descriptive')

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Medical Domain: ChestMNIST (64×64→224×224)\n"
             "ResNet18 | BiomedCLIP | 8 Private Chest X-rays",
             fontsize=14, fontweight='bold', y=0.99)

# Row 1: Baseline
ax = axes[0, 0]
img = mpimg.imread(os.path.join(RESULTS, 'medical_baseline', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[0, 1]
img = mpimg.imread(os.path.join(RESULTS, 'medical_baseline', 'final_rec.jpg'))
ax.imshow(img)
title = "Baseline Reconstruction"
if med_baseline_m:
    title += f"\nLPIPS={med_baseline_m.get('lpips', 0):.3f} | PSNR={med_baseline_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Row 2: Pneumonia query
ax = axes[1, 0]
img = mpimg.imread(os.path.join(RESULTS, 'medical_pneumonia_descriptive', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[1, 1]
img = mpimg.imread(os.path.join(RESULTS, 'medical_pneumonia_descriptive', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "chest X-ray with bilateral patchy opacities..."'
if med_pneumonia_m:
    title += f"\nLPIPS={med_pneumonia_m.get('lpips', 0):.3f} | PSNR={med_pneumonia_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Row 3: Effusion query
ax = axes[2, 0]
img = mpimg.imread(os.path.join(RESULTS, 'medical_effusion_descriptive', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[2, 1]
img = mpimg.imread(os.path.join(RESULTS, 'medical_effusion_descriptive', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "chest X-ray with blunted costophrenic angle..."'
if med_effusion_m:
    title += f"\nLPIPS={med_effusion_m.get('lpips', 0):.3f} | PSNR={med_effusion_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Add row labels on the left
for i, label in enumerate(["Baseline GIA", "Query: Pneumonia", "Query: Effusion"]):
    axes[i, 0].text(-0.02, 0.5, label, transform=axes[i, 0].transAxes,
                     rotation=90, va='center', ha='right',
                     fontsize=11, fontweight='bold', color='#333333')

plt.tight_layout(rect=[0.03, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, 'demo_medical.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: demo_medical.png")

# =============================================================================
# Figure 3: UAV Domain (UAVScenes + CLIP)
# =============================================================================
print("\n=== Figure 3: UAV Domain ===")

uav_baseline_m = load_metrics('uav_baseline')
uav_solar_m = load_metrics('uav_solar_panels')
uav_pool_m = load_metrics('uav_swimming_pool')

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("UAV Domain: UAVScenes Drone Imagery (224×224)\n"
             "ResNet18 | CLIP ViT-L/14 | 8 Private Aerial Images",
             fontsize=14, fontweight='bold', y=0.99)

# Row 1: Baseline
ax = axes[0, 0]
img = mpimg.imread(os.path.join(RESULTS, 'uav_baseline', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[0, 1]
img = mpimg.imread(os.path.join(RESULTS, 'uav_baseline', 'final_rec.jpg'))
ax.imshow(img)
title = "Baseline Reconstruction"
if uav_baseline_m:
    title += f"\nLPIPS={uav_baseline_m.get('lpips', 0):.3f} | PSNR={uav_baseline_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Row 2: Solar panels query
ax = axes[1, 0]
img = mpimg.imread(os.path.join(RESULTS, 'uav_solar_panels', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[1, 1]
img = mpimg.imread(os.path.join(RESULTS, 'uav_solar_panels', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "Solar panels on rooftops"'
if uav_solar_m:
    title += f"\nLPIPS={uav_solar_m.get('lpips', 0):.3f} | PSNR={uav_solar_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Row 3: Swimming pool query
ax = axes[2, 0]
img = mpimg.imread(os.path.join(RESULTS, 'uav_swimming_pool', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[2, 1]
img = mpimg.imread(os.path.join(RESULTS, 'uav_swimming_pool', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "Swimming pool"'
if uav_pool_m:
    title += f"\nLPIPS={uav_pool_m.get('lpips', 0):.3f} | PSNR={uav_pool_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

for i, label in enumerate(["Baseline GIA", "Query: Solar Panels", "Query: Swimming Pool"]):
    axes[i, 0].text(-0.02, 0.5, label, transform=axes[i, 0].transAxes,
                     rotation=90, va='center', ha='right',
                     fontsize=11, fontweight='bold', color='#333333')

plt.tight_layout(rect=[0.03, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, 'demo_uav.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: demo_uav.png")

# =============================================================================
# Figure 4: Cross-Domain Comparison (best result from each domain)
# =============================================================================
print("\n=== Figure 4: Cross-Domain Comparison ===")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Cross-Domain Comparison: Ground Truth vs. Best Geminio Reconstruction\n"
             "ImageNet (CLIP) | ChestMNIST (BiomedCLIP) | UAVScenes (CLIP)",
             fontsize=14, fontweight='bold', y=0.99)

# Row 1: ImageNet
ax = axes[0, 0]
img = mpimg.imread(os.path.join(ASSETS, 'original.jpg'))
ax.imshow(img)
ax.set_title("ImageNet: 128 Private Images", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[0, 1]
img = mpimg.imread(os.path.join(ASSETS, 'Any_human_faces.jpg'))
ax.imshow(img)
ax.set_title('Geminio: "Any human faces"', fontsize=11)
ax.axis('off')

# Row 2: Medical
ax = axes[1, 0]
img = mpimg.imread(os.path.join(RESULTS, 'medical_pneumonia_descriptive', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("ChestMNIST: 8 Private X-rays", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[1, 1]
img = mpimg.imread(os.path.join(RESULTS, 'medical_pneumonia_descriptive', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "Pneumonia" (descriptive prompt)'
if med_pneumonia_m:
    title += f"\nLPIPS={med_pneumonia_m.get('lpips', 0):.3f} | PSNR={med_pneumonia_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

# Row 3: UAV
ax = axes[2, 0]
img = mpimg.imread(os.path.join(RESULTS, 'uav_solar_panels', 'a_truth.jpg'))
ax.imshow(img)
ax.set_title("UAVScenes: 8 Private Drone Images", fontsize=11, fontweight='bold')
ax.axis('off')

ax = axes[2, 1]
img = mpimg.imread(os.path.join(RESULTS, 'uav_solar_panels', 'final_rec.jpg'))
ax.imshow(img)
title = 'Geminio: "Solar panels on rooftops"'
if uav_solar_m:
    title += f"\nLPIPS={uav_solar_m.get('lpips', 0):.3f} | PSNR={uav_solar_m.get('psnr', 0):.1f} dB"
ax.set_title(title, fontsize=10)
ax.axis('off')

for i, label in enumerate(["ImageNet", "Medical", "UAV"]):
    axes[i, 0].text(-0.02, 0.5, label, transform=axes[i, 0].transAxes,
                     rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', color='#333333')

plt.tight_layout(rect=[0.03, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, 'demo_cross_domain.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print(f"  Saved: demo_cross_domain.png")

# =============================================================================
# Figure 5: Batch Composition Ablation (with vs without target)
# =============================================================================
print("\n=== Figure 5: Batch Composition Ablation ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Batch Composition Ablation: Target Present vs. Absent\n"
             "Medical (Pneumonia) | Geminio Model | Batch Size = 8",
             fontsize=14, fontweight='bold', y=0.99)

med_with_m = load_metrics('medical_pneumonia_descriptive_batch_with_geminio')
med_without_m = load_metrics('medical_pneumonia_descriptive_batch_without_geminio')

# WITH target
with_truth = os.path.join(RESULTS, 'medical_pneumonia_descriptive_batch_with_geminio', 'a_truth.jpg')
with_rec = os.path.join(RESULTS, 'medical_pneumonia_descriptive_batch_with_geminio', 'final_rec.jpg')

if os.path.exists(with_truth):
    axes[0, 0].imshow(mpimg.imread(with_truth))
    axes[0, 0].set_title("Ground Truth (WITH Pneumonia)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

if os.path.exists(with_rec):
    axes[0, 1].imshow(mpimg.imread(with_rec))
    title = "Reconstruction (WITH Pneumonia)"
    if med_with_m:
        title += f"\nLPIPS={med_with_m.get('lpips', 0):.3f} | Attack F1={med_with_m.get('attack_f1', 0):.3f}"
    axes[0, 1].set_title(title, fontsize=10)
    axes[0, 1].axis('off')

# WITHOUT target
without_truth = os.path.join(RESULTS, 'medical_pneumonia_descriptive_batch_without_geminio', 'a_truth.jpg')
without_rec = os.path.join(RESULTS, 'medical_pneumonia_descriptive_batch_without_geminio', 'final_rec.jpg')

if os.path.exists(without_truth):
    axes[1, 0].imshow(mpimg.imread(without_truth))
    axes[1, 0].set_title("Ground Truth (WITHOUT Pneumonia)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

if os.path.exists(without_rec):
    axes[1, 1].imshow(mpimg.imread(without_rec))
    title = "Reconstruction (WITHOUT Pneumonia)"
    if med_without_m:
        title += f"\nLPIPS={med_without_m.get('lpips', 0):.3f} | Attack F1={med_without_m.get('attack_f1', 0):.3f}"
    axes[1, 1].set_title(title, fontsize=10)
    axes[1, 1].axis('off')

for i, label in enumerate(["Target\nPresent", "Target\nAbsent"]):
    axes[i, 0].text(-0.02, 0.5, label, transform=axes[i, 0].transAxes,
                     rotation=90, va='center', ha='right',
                     fontsize=11, fontweight='bold',
                     color='green' if i == 0 else 'red')

plt.tight_layout(rect=[0.04, 0, 1, 0.94])
fig.savefig(os.path.join(OUTDIR, 'demo_batch_composition.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  Saved: demo_batch_composition.png")

print(f"\nAll demonstration figures saved to: {OUTDIR}/")
print("Files:")
for f in sorted(os.listdir(OUTDIR)):
    if f.startswith('demo_'):
        size = os.path.getsize(os.path.join(OUTDIR, f))
        print(f"  {f} ({size/1024:.0f} KB)")
