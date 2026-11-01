# Per-sample statistics of the private batch

Uniform gradient share would be 1/8 = 0.125; the two query-matched images together would hold 0.250. `own prob` is the probability the head assigns to the image own label (mean over its positive labels for the multi-label UAV case).

## medical

| encoder | head | target share | alt-class share | target own prob | other own prob | target loss | other loss |
|---|---|---|---|---|---|---|---|
| clip | rigged | 0.117 | 0.110 | 0.0016 | 0.0038 | 6.489 | 25.209 |
| clip | plain | 0.243 | 0.066 | 0.0027 | 0.1098 | 5.927 | 2.885 |
| clip | swap | 0.207 | 0.142 | 0.0001 | 0.0252 | 11.076 | 100.211 |
| simclip2 | rigged | 0.152 | 0.045 | 0.0014 | 0.0486 | 6.593 | 33.838 |
| simclip2 | plain | 0.285 | 0.091 | 0.0017 | 0.0976 | 6.397 | 2.835 |
| simclip2 | swap | 0.189 | 0.233 | 0.0005 | 0.0844 | 8.402 | 72.121 |
| simclip4 | rigged | 0.171 | 0.050 | 0.0032 | 0.0488 | 5.749 | 25.368 |
| simclip4 | plain | 0.295 | 0.084 | 0.0014 | 0.0913 | 6.601 | 2.869 |
| simclip4 | swap | 0.233 | 0.252 | 0.0003 | 0.1278 | 9.006 | 64.110 |
| fare2 | rigged | 0.096 | 0.027 | 0.0009 | 0.0741 | 7.075 | 101.168 |
| fare2 | plain | 0.273 | 0.088 | 0.0017 | 0.1153 | 6.432 | 2.707 |
| fare2 | swap | 0.111 | 0.362 | 0.0031 | 0.0224 | 5.810 | 65.778 |
| fare4 | rigged | 0.269 | 0.074 | 0.0005 | 0.0371 | 7.724 | 17.977 |
| fare4 | plain | 0.302 | 0.080 | 0.0011 | 0.0910 | 6.915 | 2.873 |
| fare4 | swap | 0.134 | 0.319 | 0.0015 | 0.1509 | 6.637 | 75.070 |
| tecoa2 | rigged | 0.114 | 0.028 | 0.0031 | 0.0663 | 5.818 | 45.215 |
| tecoa2 | plain | 0.304 | 0.072 | 0.0008 | 0.1090 | 7.127 | 2.839 |
| tecoa2 | swap | 0.071 | 0.511 | 0.0015 | 0.0469 | 10.716 | 289.530 |
| tecoa4 | rigged | 0.122 | 0.033 | 0.0011 | 0.0512 | 6.801 | 61.250 |
| tecoa4 | plain | 0.303 | 0.072 | 0.0010 | 0.0849 | 6.970 | 2.945 |
| tecoa4 | swap | 0.116 | 0.339 | 0.0049 | 0.0888 | 6.933 | 144.590 |

## uav

| encoder | head | target share | alt-class share | target own prob | other own prob | target loss | other loss |
|---|---|---|---|---|---|---|---|
| clip | rigged | 0.693 | 0.162 | 0.1429 | 0.6917 | 535.965 | 18.435 |
| clip | plain | 0.152 | 0.370 | 0.8908 | 0.8108 | 1.774 | 14.303 |
| clip | swap | 0.080 | 0.663 | 0.8665 | 0.8266 | 2.555 | 264.865 |
| simclip2 | rigged | 0.548 | 0.211 | 0.1429 | 0.7472 | 221.698 | 13.848 |
| simclip2 | plain | 0.161 | 0.390 | 0.8823 | 0.8092 | 1.722 | 13.358 |
| simclip2 | swap | 0.063 | 0.413 | 0.8004 | 0.6823 | 3.260 | 419.671 |
| simclip4 | rigged | 0.415 | 0.388 | 0.2856 | 0.7484 | 205.868 | 39.718 |
| simclip4 | plain | 0.205 | 0.332 | 0.7949 | 0.8080 | 3.595 | 13.065 |
| simclip4 | swap | 0.109 | 0.289 | 0.7025 | 0.6444 | 4.755 | 279.157 |
| fare2 | rigged | 0.596 | 0.191 | 0.1429 | 0.7585 | 264.140 | 13.135 |
| fare2 | plain | 0.134 | 0.378 | 0.9378 | 0.8172 | 1.040 | 11.109 |
| fare2 | swap | 0.062 | 0.407 | 0.7835 | 0.6804 | 3.615 | 515.460 |
| fare4 | rigged | 0.410 | 0.268 | 0.3085 | 0.6836 | 255.965 | 26.330 |
| fare4 | plain | 0.169 | 0.334 | 0.8639 | 0.7952 | 1.996 | 13.594 |
| fare4 | swap | 0.132 | 0.297 | 0.6476 | 0.6820 | 6.170 | 236.115 |
| tecoa2 | rigged | 0.371 | 0.292 | 0.5967 | 0.7807 | 46.853 | 12.604 |
| tecoa2 | plain | 0.114 | 0.350 | 0.9119 | 0.8285 | 1.180 | 14.322 |
| tecoa2 | swap | 0.061 | 0.449 | 0.7983 | 0.6410 | 3.584 | 525.236 |
| tecoa4 | rigged | 0.600 | 0.207 | 0.2857 | 0.7765 | 328.795 | 14.151 |
| tecoa4 | plain | 0.111 | 0.357 | 0.9099 | 0.8152 | 1.200 | 17.049 |
| tecoa4 | swap | 0.174 | 0.282 | 0.5471 | 0.6156 | 10.217 | 197.430 |


## Reading (2026-10-31)

Two of the 8 images are query-matched, so a head that treats the batch uniformly puts 0.250 of the
last-layer gradient on them. Averaged over the 7 encoders:

| domain | head | target gradient share | encoders above uniform | target/other loss ratio on this batch |
|---|---|---|---|---|
| UAV | plain | 0.150 (0.111–0.205) | 0/7 | 0.13x |
| UAV | rigged | **0.519** (0.371–0.693) | **7/7** | **15.29x** |
| UAV | swap | 0.097 (0.061–0.174) | 0/7 | 0.02x |
| medical | plain | 0.287 (0.243–0.304) | 6/7 | 2.32x |
| medical | rigged | **0.149** (0.096–0.269) | **1/7** | **0.20x** |
| medical | swap | 0.151 (0.071–0.233) | 0/7 | 0.09x |

**UAV: the assumed mechanism is confirmed directly.** Under the rigged head the two solar images carry
roughly half the last-layer gradient, against 0.15 under an honest head, and their own-label
probability drops to 0.14–0.60 while the rest of the batch stays at 0.68–0.78. Low confidence on the
matched images and concentrated gradient appear together, as the account predicts, and neither appears
without the rigged head. Swapping the query moves the concentration to the airstrip images
(share 0.28–0.66) and leaves the solar images below uniform. So the head selects, the query decides
what, and the effect is visible in the gradient before any reconstruction is attempted.

**Medical: the mechanism never engages, which explains the domain null.** Under the rigged head the two
pneumonia images carry *less* gradient than uniform on 6 of 7 encoders, and less than they carry under
an honest head (0.149 against 0.287). On this batch their loss is a fifth of the other images', not a
multiple of it, despite the head showing an 8–14x loss ratio on the auxiliary set. The head is
amplifying something, but not the pneumonia images.

This completes a chain that was previously inferred. Control B found that the attacker guides rank
pneumonia at or below chance on the auxiliary set (AUROC 0.38–0.49). A head trained to amplify whatever
the guide scores highly therefore learns to amplify a non-pneumonia subset, and on the private batch the
targets get no amplification. That is a sufficient explanation for every downstream medical result: the
near-zero selectivity, the reconstructions that never beat a random draw, and the apparent floor. It
also means the medical null is a property of the guide's retrieval on this dataset, not evidence that
the attack is harmless on medical imagery in general.

Caveats: one batch per domain, single-label targets on medical, and the gradient norm is computed for
the head's last layer only, so it does not account for what the frozen encoder contributes upstream.
