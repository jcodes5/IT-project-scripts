#!/usr/bin/env python3
"""
AHP weights for the exposure layers (population, buildings, roads).

Edit MATRIX with YOUR pairwise judgements, run the script, and paste the printed
EXPOSURE_WEIGHTS line into the CONFIG block of phase5_exposure_risk.py.

Saaty scale: row i compared with column j
  1 = equally important, 3 = moderately more, 5 = strongly more, 7 = very strongly more, 9 = extremely more
  (2, 4, 6, 8 are in-between values). The lower triangle must be the reciprocal of the upper triangle.

The matrix below is a DRAFT to start the discussion, not a finished judgement.
Change any value you do not agree with and write down why: the reasons go into your methodology chapter.

Run: python ahp_exposure_weights.py
"""
import numpy as np

LAYERS = ["population", "buildings", "roads"]

# Draft judgements (row vs column):
#   population vs buildings = 2  : people come first for life safety, but modelled population is
#                                  partly built from building data, so it is only slightly ahead
#   population vs roads     = 3  : people are moderately more important than road length
#   buildings  vs roads     = 2  : buildings are assets that can be damaged; roads matter mainly
#                                  for access and evacuation
MATRIX = np.array([
    [1,   2,   3],
    [1/2, 1,   2],
    [1/3, 1/2, 1],
], dtype="float64")

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

n = len(LAYERS)
assert MATRIX.shape == (n, n), "MATRIX size must match LAYERS"
assert np.allclose(MATRIX, 1.0 / MATRIX.T), "MATRIX must be reciprocal (a_ij = 1 / a_ji)"

vals, vecs = np.linalg.eig(MATRIX)
k = int(np.argmax(vals.real))
lam = float(vals[k].real)
w = np.abs(vecs[:, k].real)
w = w / w.sum()
CI = (lam - n) / (n - 1)
CR = CI / RI[n] if RI[n] > 0 else 0.0
gm = np.prod(MATRIX, axis=1) ** (1.0 / n)
gm = gm / gm.sum()

print("lambda_max = %.4f   CI = %.4f   CR = %.4f (%.2f%%)" % (lam, CI, CR, 100 * CR))
print("Consistency:", "ACCEPTABLE (below 0.10)" if CR < 0.10 else "NOT ACCEPTABLE: revise the judgements")
for name, a, b in zip(LAYERS, w, gm):
    print("  %-12s weight %.3f   (geometric-mean check %.3f)" % (name, a, b))
print()
print("Paste into phase5_exposure_risk.py:")
print("EXPOSURE_WEIGHTS = {" + ", ".join('"%s": %.3f' % (nm, a) for nm, a in zip(LAYERS, w)) + ', "facilities": 0.0}')