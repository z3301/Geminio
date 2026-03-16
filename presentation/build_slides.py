"""Build Beamer-style PDF slides using matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, '..', 'assets')
FIGURES = os.path.join(BASE, 'figures')
OUTPATH = os.path.join(BASE, 'slides.pdf')

# Slide dimensions (16:9)
W, H = 16, 9

def new_slide(pdf, title=None, subtitle=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('#1a1a2e')
    if title:
        fig.text(0.5, 0.95 if not subtitle else 0.96, title,
                 ha='center', va='top', fontsize=28, fontweight='bold',
                 color='white', family='sans-serif')
    if subtitle:
        fig.text(0.5, 0.91, subtitle,
                 ha='center', va='top', fontsize=16, color='#aabbcc',
                 family='sans-serif')
    return fig

with PdfPages(OUTPATH) as pdf:

    # ===== SLIDE 1: Title =====
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('#1a1a2e')
    fig.text(0.5, 0.62, 'Geminio', ha='center', va='center',
             fontsize=52, fontweight='bold', color='white', family='sans-serif')
    fig.text(0.5, 0.50, 'Language-Guided Gradient Inversion Attacks\nin Federated Learning',
             ha='center', va='center', fontsize=22, color='#88ccff', family='sans-serif',
             linespacing=1.5)
    fig.text(0.5, 0.34, 'Phase 1 & 2 Progress Report',
             ha='center', va='center', fontsize=18, color='#cccccc', family='sans-serif')
    fig.text(0.5, 0.22, 'Dan Zimmerman\nAdvisor: Dr. Ahmed Imteaj',
             ha='center', va='center', fontsize=16, color='#999999', family='sans-serif',
             linespacing=1.5)
    fig.text(0.5, 0.10, 'Jiao et al., ICCV 2025',
             ha='center', va='center', fontsize=13, color='#666688',
             family='sans-serif', style='italic')
    pdf.savefig(fig)
    plt.close(fig)
    print("Slide 1: Title")

    # ===== SLIDE 2: Background & Method =====
    fig = new_slide(pdf, 'Background & Attack Mechanism')

    # Left side: text
    text_lines = [
        ("Federated Learning Threat Model:", True),
        ("  - Clients train on private data, share only gradients", False),
        ("  - Malicious server inverts gradients to reconstruct images", False),
        ("  - Problem: Existing GIAs fail at large batch sizes (>=64)", False),
        ("", False),
        ("Geminio's Key Contribution:", True),
        ("  - Uses CLIP (VLM) to enable natural language queries", False),
        ("     targeting specific private data", False),
        ("  - Reshapes the model's loss surface so query-matching", False),
        ("     images produce dominant gradients", False),
        ("  - Enables targeted reconstruction from 128-sample batches", False),
        ("", False),
        ("Loss Surface Reshaping:", True),
        ("  L = E[ loss_i * (1 - clip_similarity_i) ]", False),
        ("  Query-matching images -> amplified gradients", False),
        ("  Non-matching images -> suppressed gradients", False),
    ]
    y_start = 0.84
    for i, (line, is_bold) in enumerate(text_lines):
        fig.text(0.04, y_start - i * 0.045, line,
                 ha='left', va='top', fontsize=13,
                 fontweight='bold' if is_bold else 'normal',
                 color='#ffffff' if is_bold else '#cccccc',
                 family='sans-serif')

    # Right side: architecture diagram
    ax = fig.add_axes([0.55, 0.08, 0.42, 0.78])
    ax.imshow(mpimg.imread(os.path.join(ASSETS, 'intro-git.png')))
    ax.axis('off')

    pdf.savefig(fig)
    plt.close(fig)
    print("Slide 2: Background & Method")

    # ===== SLIDE 3: Results =====
    fig = new_slide(pdf, 'Experimental Results — ImageNet (128 Private Samples)')

    # Left: images grid
    images_data = [
        (os.path.join(ASSETS, 'original.jpg'), 'Ground Truth'),
        (os.path.join(ASSETS, 'baseline.jpg'), 'Baseline GIA'),
        (os.path.join(ASSETS, 'Any_jewelry.jpg'), '"Any jewelry"'),
        (os.path.join(ASSETS, 'Any_human_faces.jpg'), '"Any human faces"'),
        (os.path.join(ASSETS, 'Any_males_with_a_beard.jpg'), '"Any males with a beard"'),
        (os.path.join(ASSETS, 'Any_guns.jpg'), '"Any guns"'),
        (os.path.join(ASSETS, 'Any_females_riding_a_horse.jpg'), '"Females riding horse"'),
    ]

    positions = [
        [0.02, 0.50, 0.28, 0.35],   # Ground truth
        [0.32, 0.50, 0.28, 0.35],   # Baseline
        [0.02, 0.07, 0.17, 0.35],   # Jewelry
        [0.20, 0.07, 0.17, 0.35],   # Faces
        [0.38, 0.07, 0.17, 0.35],   # Beard
    ]

    # Top row: ground truth + baseline
    for idx in range(2):
        path, title = images_data[idx]
        ax = fig.add_axes([0.02 + idx * 0.30, 0.48, 0.28, 0.38])
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11, color='white', pad=4)
        ax.axis('off')

    # Bottom row: 5 queries
    for idx in range(5):
        path, title = images_data[idx + 2]
        ax = fig.add_axes([0.02 + idx * 0.12, 0.05, 0.11, 0.35])
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=7, color='white', pad=2)
        ax.axis('off')

    # Right side: results table
    table_lines = [
        ("Method", "Loss", "Imprv.", True),
        ("─" * 35, "", "", False),
        ("Baseline GIA", "0.1828", "—", False),
        ("─" * 35, "", "", False),
        ('"Any jewelry"', "0.0965", "47.2%", False),
        ('"Any human faces"', "0.0750", "59.0%", False),
        ('"Males w/ beard"', "0.1097", "40.0%", False),
        ('"Any guns"', "0.0984", "46.2%", False),
        ('"Female on horse"', "0.1222", "33.2%", False),
    ]

    fig.text(0.67, 0.83, "Reconstruction Loss Comparison",
             fontsize=14, fontweight='bold', color='#88ccff',
             ha='left', family='sans-serif')

    for i, (col1, col2, col3, is_header) in enumerate(table_lines):
        y = 0.76 - i * 0.042
        color = '#88ccff' if is_header else ('#ffcc44' if col1 == '"Any human faces"' else '#cccccc')
        weight = 'bold' if is_header or col1 == '"Any human faces"' else 'normal'
        fig.text(0.67, y, col1, fontsize=11, color=color, fontweight=weight,
                 family='monospace', ha='left')
        fig.text(0.84, y, col2, fontsize=11, color=color, fontweight=weight,
                 family='monospace', ha='center')
        fig.text(0.93, y, col3, fontsize=11, color=color, fontweight=weight,
                 family='monospace', ha='center')

    # Key findings
    findings = [
        "Key Findings:",
        "- Baseline: all 128 images unrecognizable",
        "- Geminio: query-matched images clearly",
        "  reconstructed; others suppressed (gray)",
        "- Best: \"human faces\" (59% lower loss)",
    ]
    for i, line in enumerate(findings):
        fig.text(0.67, 0.35 - i * 0.045, line, fontsize=11,
                 fontweight='bold' if i == 0 else 'normal',
                 color='#ffffff' if i == 0 else '#aaaaaa',
                 family='sans-serif')

    pdf.savefig(fig)
    plt.close(fig)
    print("Slide 3: Results")

    # ===== SLIDE 4: Next Steps =====
    fig = new_slide(pdf, 'Next Steps — Healthcare Application')

    sections = [
        ("Completed (Phases 1–2):", [
            "  ✓  Paper review — understood attack mechanism, VLM guidance, loss reshaping",
            "  ✓  Full pipeline reproduced on ImageNet (VLM embeddings → training → attacks)",
            "  ✓  Debugged & documented issues (dependency conflicts, missing configs)",
        ]),
        ("Phase 3 — Healthcare Dataset Application:", [
            "  •  Apply Geminio attack to medical images (MRI, X-ray, patient records)",
            "  •  Test if patient data can be exposed despite FL privacy guarantees",
            "  •  Replace CLIP with medical VLM (BiomedCLIP) for better alignment",
            "  •  Example queries: \"Any chest X-ray showing pneumonia\"",
        ]),
        ("Future Directions:", [
            "  •  Propose defense mechanisms (not covered in original paper)",
            "  •  Explore applicability to DoD / mission-critical settings",
            "  •  Differential privacy, gradient clipping, secure aggregation",
        ]),
    ]

    y = 0.84
    for header, items in sections:
        fig.text(0.06, y, header, fontsize=16, fontweight='bold',
                 color='#88ccff', family='sans-serif')
        y -= 0.05
        for item in items:
            fig.text(0.06, y, item, fontsize=13, color='#cccccc',
                     family='sans-serif')
            y -= 0.045
        y -= 0.025

    pdf.savefig(fig)
    plt.close(fig)
    print("Slide 4: Next Steps")

print(f"\nSlides saved to: {OUTPATH}")
