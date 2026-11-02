"""Registry of the medical datasets the prototype scripts can run on.

Keys are the domain names used in result-dir prefixes, head-file stems and summaries.
`medical` is the original ChestMNIST pipeline; `pathmnist` is the second medical dataset
added 2026-11-02. The attacker VLM and base query are per dataset; the target class of
`pathmnist` was fixed after the per-class retrieval scan
(results/pathmnist/retrieval_scan.md) and before any attack ran.
"""

MEDICAL = {
    'medical': dict(
        flag='chestmnist',
        embed_prefix='medical',            # data/medical-{vlm}-test.pt
        n_classes=15,
        target=6, target_name='pneumonia',
        query='Any chest X-ray showing pneumonia', tag='pna_base',
        attacker_vlm='biomedclip',
        batch='assets/medical_heldout_pna',
        head_dir='malicious_models_medical_v2',
    ),
    'pathmnist': dict(
        flag='pathmnist',
        embed_prefix='pathmnist',          # data/pathmnist-{vlm}-test.pt
        n_classes=9,
        # Fixed 2026-11-02 14:05 from results/pathmnist/retrieval_scan.md, before any attack run:
        # class 8 is the clinically sensitive class, BiomedCLIP retrieves it (AUROC 0.87, prec@10% 0.57 vs
        # base 0.17) and the CLIP-family guides spread from 0.54 (SimCLIP-4) to 0.78 (TeCoA-4).
        target=8, target_name='colorectal adenocarcinoma epithelium',
        query='Any colorectal histopathology image showing colorectal adenocarcinoma epithelium', tag='tum_base',
        attacker_vlm='biomedclip',
        batch='assets/pathmnist_heldout',
        head_dir='malicious_models_pathmnist',
    ),
}

GUIDES = ['biomedclip', 'clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']


def embed_path(domain, vlm, root='./data'):
    import os
    return os.path.join(root, f"{MEDICAL[domain]['embed_prefix']}-{vlm}-test.pt")


def is_medical(domain):
    return domain in MEDICAL
