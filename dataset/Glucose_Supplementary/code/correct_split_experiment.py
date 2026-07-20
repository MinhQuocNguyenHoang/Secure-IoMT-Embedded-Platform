# -*- coding: utf-8 -*-
"""Fix the split the RIGHT way and measure honest accuracy.
   (A) LEAKY   augment -> then split      (the original pipeline)
   (B) FIXED   split real FIRST, augment ONLY train  (the correct fix)
   (C) STRICT  Leave-One-Session-Out      (session-aware, deployment-realistic)
   Done for both the 3 entropy features and the baseline-corrected notch feature."""
import numpy as np, pandas as pd, warnings, glob, os
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

rng = np.random.default_rng(0)
EDIR = "Glucose_Data_extracted/Glucose_Data"
FE = ["Logmag_S11_S", "Logmag_S11_A", "Logmag_S11_D"]
ONE = ["Label1", "Label2", "Label3", "Label4"]

def jitter_augment(X, y, k=10, scale=0.3):
    """add k-1 Gaussian-jittered copies per row (train only)."""
    sd = X.std(0) * scale + 1e-9
    Xs = [X]; ys = [y]
    for _ in range(k - 1):
        Xs.append(X + rng.normal(0, sd, X.shape)); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

def models():
    return {"LDA": make_pipeline(StandardScaler(), LDA()),
            "MLP 256-128": make_pipeline(StandardScaler(), MLPClassifier((256,128), activation="tanh", max_iter=3000, random_state=0))}

# ---------------- ENTROPY data (real, per session) ----------------
def load_entropy_set(st):
    fr = []
    for l in [1,2,3,4]:
        d = pd.read_excel(f"{EDIR}/Entropy data/notemp/Logmag_S11_Label{l}_set{st}.xlsx").drop_duplicates(subset=FE).copy()
        d["y"] = l-1; d["sess"] = st; fr.append(d)
    return pd.concat(fr, ignore_index=True)
E = pd.concat([load_entropy_set(1), load_entropy_set(2)], ignore_index=True)
Xe, ye, se = E[FE].values, E["y"].values, E["sess"].values

# (A) LEAKY for entropy: augment->split, from the combined augmented files
comb = pd.concat([pd.read_excel(f"{EDIR}/Augment/ผสมset1set2ทั้งหมดที่แบ่งไปใช้ในเปเปอร์และCode/Logmag_S11_Label{l}_combined.xlsx") for l in [1,2,3,4]], ignore_index=True)
Xc, yc = comb[FE].values, comb[ONE].values.argmax(1)

def fixed_cv(X, y, aug=True):
    out = {}
    for nm, _ in models().items():
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
            Xtr, ytr = (jitter_augment(X[tr], y[tr]) if aug else (X[tr], y[tr]))
            m = models()[nm].fit(Xtr, ytr); accs.append(accuracy_score(y[te], m.predict(X[te])))
        out[nm] = np.mean(accs)
    return out

def loso(X, y, s):
    out = {}
    for nm, _ in models().items():
        a = []
        for te in [1, 2]:
            tr = s != te; ts = s == te
            m = models()[nm].fit(*jitter_augment(X[tr], y[tr])); a.append(accuracy_score(y[ts], m.predict(X[ts])))
        out[nm] = np.mean(a)
    return out

print("="*70)
print("ENTROPY features (Shannon, ApEn, Correlation Dim) — the ORIGINAL method")
print("="*70)
# A: leaky
la = {}
for nm in models():
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    m = models()[nm].fit(Xtr, ytr); la[nm] = accuracy_score(yte, m.predict(Xte))
fb = fixed_cv(Xe, ye, aug=True)
lo = loso(Xe, ye, se)
print(f"{'protocol':42s}{'LDA':>10s}{'MLP':>12s}")
print(f"{'(A) LEAKY  augment -> split (original)':42s}{la['LDA']*100:9.1f}%{la['MLP 256-128']*100:11.1f}%")
print(f"{'(B) FIXED  split first, augment train':42s}{fb['LDA']*100:9.1f}%{fb['MLP 256-128']*100:11.1f}%")
print(f"{'(C) STRICT session-aware LOSO':42s}{lo['LDA']*100:9.1f}%{lo['MLP 256-128']*100:11.1f}%")
print("   chance level = 25%")

# ---------------- NOTCH feature (real, per session) ----------------
print("\n" + "="*70)
print("NOTCH feature (baseline-corrected 4.3 GHz) — the BETTER physical feature")
print("="*70)
N = pd.read_csv("s2p_notch_features.csv")
N = N.dropna(subset=["lvl"]).copy(); N["lvl"] = N["lvl"].astype(int)
Xn, yn, sn = N[["notch_abs"]].values, N["lvl"].values, N["sess"].values
fbn = fixed_cv(Xn, yn, aug=True); lon = loso(Xn, yn, sn)
print(f"{'protocol':42s}{'LDA':>10s}{'MLP':>12s}")
print(f"{'(B) FIXED  split first, augment train':42s}{fbn['LDA']*100:9.1f}%{fbn['MLP 256-128']*100:11.1f}%")
print(f"{'(C) STRICT session-aware LOSO':42s}{lon['LDA']*100:9.1f}%{lon['MLP 256-128']*100:11.1f}%")
