# -*- coding: utf-8 -*-
"""1D-CNN on the full baseline-corrected S11 spectrum, under the SAME two protocols
   used in Table 2 (augment-before-split = leaky, and Leave-One-Session-Out = honest).
   Also re-runs the full-spectrum MLP as a sanity check that this pipeline reproduces
   the paper's numbers (~95% within-session / ~43% LOSO)."""
import os, glob, re, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

BASE = "Glucose_Data_extracted/Glucose_Data/Raw data from vna"
SESS = {1: os.path.join(BASE, "032067 เก็บข้อมูล Glucose Blood 1"),
        2: os.path.join(BASE, "032167 เก็บข้อมูล Glucose Blood 2")}

def s11_db(path):
    f, re_, im_ = [], [], []
    for line in open(path, encoding="latin-1"):
        line = line.strip()
        if not line or line[0] in "!#":
            continue
        p = line.split()
        if len(p) < 3:
            continue
        try:
            f.append(float(p[0])); re_.append(float(p[1])); im_.append(float(p[2]))
        except ValueError:
            continue
    return 20 * np.log10(np.hypot(np.array(re_), np.array(im_)))

def level(c):
    if 45 <= c <= 65:   return 0
    if 70 <= c <= 110:  return 1
    if 125 <= c <= 200: return 2   # matches original experiments (multi_notch_experiment.py); 120 excluded
    if 220 <= c <= 300: return 3
    return -1  # 0 mg/dL reference or 120 (boundary) -> excluded

# ---- build per-session baseline-corrected spectra ----
def load_session(d):
    specs, concs = [], []
    for p in glob.glob(os.path.join(d, "*.s2p")):
        b = os.path.basename(p)
        if "Air" in b:
            continue
        m = re.search(r"Glu(\d+)", b)
        if not m:
            continue
        db = s11_db(p)
        if len(db) != 101:
            continue
        specs.append(db); concs.append(int(m.group(1)))
    specs = np.array(specs); concs = np.array(concs)
    ref = specs[concs == 0].mean(0)            # per-session 0 mg/dL reference
    dS = specs - ref                            # baseline-corrected dS11 spectrum
    keep = np.array([level(c) for c in concs])
    mask = keep >= 0
    return dS[mask], keep[mask]

X1, y1 = load_session(SESS[1]); X2, y2 = load_session(SESS[2])
print(f"session1: {X1.shape} levels={np.bincount(y1)} | session2: {X2.shape} levels={np.bincount(y2)}")

def jitter(X, y, k=10, scale=0.3, rng=None):
    rng = rng or np.random.default_rng(0)
    sd = X.std(0) * scale + 1e-9
    Xs, ys = [X], [y]
    for _ in range(k - 1):
        Xs.append(X + rng.normal(0, sd, X.shape)); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

def make_cnn(seed=0):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((101, 1)),
        tf.keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def fit_cnn(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr)[..., None], sc.transform(Xte)[..., None]
    m = make_cnn(seed)
    es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)
    m.fit(Xtr, ytr, epochs=120, batch_size=32, verbose=0, callbacks=[es])
    return accuracy_score(yte, m.predict(Xte, verbose=0).argmax(1))

def fit_mlp(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier((256, 128), activation="tanh", max_iter=4000, random_state=0)
    m.fit(sc.transform(Xtr), ytr)
    return accuracy_score(yte, m.predict(sc.transform(Xte)))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def fit_lda(Xtr, ytr, Xte, yte):
    m = make_pipeline_local(Xtr); m.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))
def make_pipeline_local(Xtr):
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LDA())

SEEDS = [0, 1, 2]
rng = np.random.default_rng(42)

# ===== Protocol A: LEAKY (augment-before-split, random 80/20) =====
Xall = np.vstack([X1, X2]); yall = np.concatenate([y1, y2])
Xa, ya = jitter(Xall, yall, rng=rng)
idx = rng.permutation(len(Xa)); cut = int(0.8 * len(Xa))
tr, te = idx[:cut], idx[cut:]
cnn_leaky = [fit_cnn(Xa[tr], ya[tr], Xa[te], ya[te], s) for s in SEEDS]
mlp_leaky = fit_mlp(Xa[tr], ya[tr], Xa[te], ya[te])

# ===== Protocol C: LOSO (augment train only, test held-out session clean) =====
def loso(fitfn, **kw):
    accs = []
    for (Xtr, ytr), (Xte, yte) in [((X1, y1), (X2, y2)), ((X2, y2), (X1, y1))]:
        Xt, yt = jitter(Xtr, ytr, rng=np.random.default_rng(7))
        accs.append(fitfn(Xt, yt, Xte, yte, **kw) if "seed" in kw or fitfn is fit_cnn
                    else fitfn(Xt, yt, Xte, yte))
    return np.mean(accs)

cnn_loso = [np.mean([
    fit_cnn(*[*jitter(Xtr, ytr, rng=np.random.default_rng(7))], Xte, yte, s)
    for (Xtr, ytr), (Xte, yte) in [((X1, y1), (X2, y2)), ((X2, y2), (X1, y1))]
]) for s in SEEDS]
mlp_loso = np.mean([
    fit_mlp(*jitter(Xtr, ytr, rng=np.random.default_rng(7)), Xte, yte)
    for (Xtr, ytr), (Xte, yte) in [((X1, y1), (X2, y2)), ((X2, y2), (X1, y1))]
])

# ===== reference: NO-augmentation honest protocols (matches multi_notch_experiment.py that produced the paper's 43%) =====
from sklearn.model_selection import StratifiedKFold
def loso_noaug(fitfn):
    return np.mean([fitfn(Xtr, ytr, Xte, yte)
        for (Xtr, ytr), (Xte, yte) in [((X1, y1), (X2, y2)), ((X2, y2), (X1, y1))]])
def cv_noaug(fitfn):  # pooled 5-fold within-distribution
    accs = []
    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=1).split(Xall, yall):
        accs.append(fitfn(Xall[itr], yall[itr], Xall[ite], yall[ite]))
    return np.mean(accs)
lda_loso_noaug = loso_noaug(fit_lda); mlp_loso_noaug = loso_noaug(fit_mlp)
lda_cv = cv_noaug(fit_lda);           mlp_cv = cv_noaug(fit_mlp)
lda_leaky = fit_lda(Xa[tr], ya[tr], Xa[te], ya[te])
lda_loso = loso(fit_lda)

def pct(a): return f"{100*np.mean(a):.1f}%" + (f" (±{100*np.std(a):.1f})" if hasattr(a, '__len__') else "")
print("\n========= RESULTS (full 101-pt dS11 spectrum, Level-3 = 125-200) =========")
print(f"{'model':12s}{'leaky aug->split':18s}{'pooled 5-fold CV':18s}{'LOSO (no aug)':16s}{'LOSO (train-aug)':16s}")
print(f"{'LDA':12s}{pct(lda_leaky):18s}{pct(lda_cv):18s}{pct(lda_loso_noaug):16s}{pct(lda_loso):16s}  <- reproduces paper full-spectrum?")
print(f"{'MLP 256-128':12s}{pct(mlp_leaky):18s}{pct(mlp_cv):18s}{pct(mlp_loso_noaug):16s}{pct(mlp_loso):16s}")
print(f"{'1D-CNN':12s}{pct(cnn_leaky):18s}{'(train-aug)':18s}{'--':16s}{pct(cnn_loso):16s}  <- NEW")
print("chance = 25%  | paper full-spectrum: ~91-97% within / 43% LOSO")
