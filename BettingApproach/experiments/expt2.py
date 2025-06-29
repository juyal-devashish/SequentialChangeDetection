import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))



import numpy as np
from data_source import OneHotSource, ChangePointSource
from change_detector import ChangeDetector
from sequential_test import SequentialHypothesisTestONS, SequentialHypothesisTest, IdentityClassifier

my_trained_clf = IdentityClassifier()
# (A) Choose your test constructor: either the original...
from sequential_test import SequentialHypothesisTest
def make_test_grid():
    return SequentialHypothesisTest(
        classifier=my_trained_clf,   # e.g. loaded or trained elsewhere
        n_classes=K,
        alpha=0.001,
        M=200,
        smoothing=0.5,
        N_max=500
    )

# ...or the ONS-based version:
def make_test_ons():
    return SequentialHypothesisTestONS(
        classifier=my_trained_clf,
        n_classes=K,
        alpha=0.001,
        eta=1.0,
        reg=1e-2,
        N_max=500
    )

# (B) Define your pre- and post-change distributions
p = np.array([0.6, 0.4])      # e.g. two-class pre-change
q = np.array([0.4, 0.6])      # post-change flips probabilities
K = len(p)
# Stream lengths
TOTAL_LEN = 1000
TC = 200

# Create a change-point source
pre_src  = OneHotSource(p, length=TOTAL_LEN)
post_src = OneHotSource(q, length=TOTAL_LEN)
cp_source = ChangePointSource(
    pre_source=pre_src,
    post_source=post_src,
    T_c=TC,
    length=TOTAL_LEN,
    randomize_Tc=False
)

# (C) Instantiate the detector (pick grid or ONS)
detector = ChangeDetector(test_ctor=make_test_ons, N_max=500, W=200)

# (D) Run and report
alarm, t_hat = detector.run_on_source(cp_source)
if alarm:
    print(f"→ Change detected at time t = {t_hat} (true T_c = {TC})")
else:
    print("→ No alarm within stream length.")
