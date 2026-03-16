"""Build the written summary PDF using fpdf2."""
import os
from fpdf import FPDF

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE, 'figures')
OUTPATH = os.path.join(BASE, 'summary.pdf')


class SummaryPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Geminio Project: Phase 1 & 2 Summary Report', align='R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(25, 55, 95)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 55, 95)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, '-')
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet_bold_start(self, bold_part, rest, indent=10):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.cell(5, 5.5, '-')
        self.set_font('Helvetica', 'B', 10)
        self.write(5.5, bold_part)
        self.set_font('Helvetica', '', 10)
        self.write(5.5, rest)
        self.ln(6.5)


pdf = SummaryPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 20)
pdf.set_text_color(25, 55, 95)
pdf.cell(0, 12, 'Geminio Project', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, 'Phase 1 & 2 Summary Report', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 6, 'Dan Zimmerman  |  Advisor: Dr. Ahmed Imteaj', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# === Phase 1 ===
pdf.section_title('Phase 1: Paper Review')

pdf.body_text(
    'The ICCV 2025 paper "Geminio: Language-Guided Gradient Inversion Attacks in Federated '
    'Learning" by Jiao et al. introduces a novel attack that uses Vision-Language Models (VLMs) '
    'to enable targeted gradient inversion in federated learning.'
)

pdf.subsection_title('Key Concepts')

pdf.bullet_bold_start(
    'Federated Learning & Gradient Inversion: ',
    'In FL, clients keep data private and share only model gradients. Gradient Inversion '
    'Attacks (GIAs) reconstruct private images from these gradients. However, existing GIAs '
    'fail when the client batch size is large (>=64 images) because the gradient signal from '
    'individual images becomes too diluted.'
)

pdf.bullet_bold_start(
    'Geminio\'s Approach: ',
    'A malicious FL server uses CLIP (a vision-language model) to compute similarity between '
    'each training image and a natural language query (e.g., "Any human faces"). The server '
    'trains a malicious model whose loss surface amplifies gradients from query-matching images '
    'while suppressing others.'
)

pdf.bullet_bold_start(
    'Loss Surface Reshaping: ',
    'The training objective L = E[loss_i * (1 - p_i)] where p_i is the CLIP similarity '
    'probability. This causes query-matching images to produce dominant gradients during '
    'the FL training round.'
)

pdf.bullet_bold_start(
    'Task-Agnostic Targeting: ',
    'The same framework supports arbitrary natural language queries -from objects ("jewelry") '
    'to people ("human faces") to complex scenes ("females riding a horse").'
)

# === Phase 2 ===
pdf.section_title('Phase 2: Code Implementation')

pdf.subsection_title('Pipeline Overview')

pdf.body_text(
    'The implementation consists of three stages, all successfully executed on the lab\'s DGX '
    'system with NVIDIA H200 GPUs:'
)

pdf.bullet_bold_start(
    'VLM Embedding Preprocessing: ',
    'Generated CLIP (ViT-L/14) embeddings for 50,000 ImageNet validation images and 1,000 '
    'class text descriptions. (~22 minutes on 1 GPU)'
)

pdf.bullet_bold_start(
    'Malicious Model Training: ',
    'Trained 5 query-specific malicious ResNet34 classifier heads (5 epochs each, Adam '
    'optimizer). Parallelized across 5 GPUs. (~5 minutes total)'
)

pdf.bullet_bold_start(
    'Gradient Inversion Attacks: ',
    'Ran HFGradInv attack (24,000 iterations) for each of the 5 queries plus a baseline. '
    'Parallelized across 6 GPUs. (~45 minutes total)'
)

pdf.subsection_title('Issues Encountered and Resolved')

pdf.bullet_bold_start(
    'Dependency conflict: ',
    'Latest transformers library was incompatible with PyTorch 2.1.2. '
    'Fixed by pinning transformers==4.36.2.'
)

pdf.bullet_bold_start(
    'Data path mismatch: ',
    'ImageNet archives were in a subdirectory; the code expected them at a different path. '
    'Fixed with symlinks.'
)

pdf.bullet_bold_start(
    'Missing config file: ',
    'The Hydra configuration system required a GeminioImageNet.yaml absent from the repo. '
    'Created it based on the existing ImageNet.yaml.'
)

pdf.bullet_bold_start(
    'Class name format bug: ',
    'The embedding script failed because dataset.classes returned tuples instead of strings. '
    'Fixed by extracting the first element.'
)

pdf.subsection_title('Results')

# Results table
pdf.set_font('Helvetica', 'B', 10)
pdf.set_fill_color(25, 55, 95)
pdf.set_text_color(255, 255, 255)
col_widths = [80, 35, 45]
headers = ['Method / Query', 'Rec. Loss', 'Improvement']
for w, h in zip(col_widths, headers):
    pdf.cell(w, 7, h, border=1, align='C', fill=True)
pdf.ln()

pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(30, 30, 30)
rows = [
    ('Baseline GIA (no query)', '0.1828', '---'),
    ('Geminio: "Any jewelry"', '0.0965', '47.2%'),
    ('Geminio: "Any human faces"', '0.0750', '59.0%'),
    ('Geminio: "Males w/ beard"', '0.1097', '40.0%'),
    ('Geminio: "Any guns"', '0.0984', '46.2%'),
    ('Geminio: "Females on horse"', '0.1222', '33.2%'),
]
for i, (c1, c2, c3) in enumerate(rows):
    fill = i % 2 == 1
    if fill:
        pdf.set_fill_color(240, 245, 250)
    if 'human faces' in c1:
        pdf.set_font('Helvetica', 'B', 10)
    else:
        pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_widths[0], 6, c1, border=1, fill=fill)
    pdf.cell(col_widths[1], 6, c2, border=1, align='C', fill=fill)
    pdf.cell(col_widths[2], 6, c3, border=1, align='C', fill=fill)
    pdf.ln()

pdf.ln(4)

pdf.body_text(
    'The baseline attack produces entirely unrecognizable images across all 128 samples. '
    'In contrast, Geminio successfully reconstructs identifiable images for samples matching '
    'each query, while non-matching samples appear as suppressed gray patches -exactly as '
    'predicted by the paper.'
)

# Insert figure
fig_path = os.path.join(FIGURES, 'baseline_vs_geminio.png')
if os.path.exists(fig_path):
    img_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.image(fig_path, x=pdf.l_margin, w=img_w)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5,
             'Figure 1: Ground truth (left), baseline reconstruction (center), and Geminio '
             'with query "Any human faces" (right).',
             align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

# === Next Steps ===
pdf.section_title('Next Steps: Phase 3')

pdf.bullet_bold_start(
    'Healthcare dataset application: ',
    'Apply the attack to medical images (MRI, X-ray) to test whether patient data '
    'can be exposed through gradient inversion despite FL privacy guarantees.'
)

pdf.bullet_bold_start(
    'VLM adaptation: ',
    'Standard CLIP is trained on natural images; a medical VLM (e.g., BiomedCLIP, MedCLIP) '
    'may be needed for better image-text alignment on medical data.'
)

pdf.bullet_bold_start(
    'Defense exploration: ',
    'The original paper proposes no defenses. Future work could explore countermeasures such '
    'as gradient clipping, differential privacy, or secure aggregation.'
)

pdf.output(OUTPATH)
print(f"Summary saved to: {OUTPATH}")
