# -*- coding: utf-8 -*-
"""Multi-notch fusion: does combining several baseline-corrected resonant notches
   beat the single 4.3 GHz notch WITHOUT overfitting session drift?
   Honest protocols: (B) split-first 5-fold CV, (C) Leave-One-Session-Out."""
import numpy as np, glob, re, os, warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

RAW = "Glucose_Data_extracted/Glucose_Data/Raw data from vna"
SESS = {1: "032067 เก็บข้อมูล Glucose Blood 1", 2: "032167 เก็บข้อมูล Glucose Blood 2"}

def read_s2p(path):
    fr, s11, fmt, unit = [], [], "RI", "Hz"
    for line in open(path, encoding="latin-1"):
        s = line.strip()
        if not s or s.startswith("!"): continue
        if s.startswith("#"):
            t = s.lower().split(); unit = {"ghz":"GHz","mhz":"MHz","khz":"kHz","hz":"Hz"}.get(next((x for x in t if x in("ghz","mhz","khz","hz")),"hz")); fmt = "MA" if "ma" in t else "RI"; continue
        q = s.split()
        if len(q) < 3: continue
        try: v = [float(x) for x in q]
        except ValueError: continue
        fr.append(v[0]); s11.append(complex(v[1],v[2]) if fmt=="RI" else v[1]*np.exp(1j*np.deg2rad(v[2])))
    m = {"Hz":1,"kHz":1e3,"MHz":1e6,"GHz":1e9}[unit]
    return np.array(fr)*m, 20*np.log10(np.abs(np.array(s11))+1e-15)

conc = lambda fn:(int(re.search(r"Glu(\d+)",fn).group(1)) if re.search(r"Glu(\d+)",fn) else None)
def level(c):
    if 45<=c<=65:return 0
    if 70<=c<=110:return 1
    if 125<=c<=200:return 2
    if 220<=c<=300:return 3
    return None

# build per-session spectra + baselines
spectra = {}   # sess -> list of (conc, s)
base = {}
f0 = None
for sess, folder in SESS.items():
    files = sorted(glob.glob(os.path.join(RAW, folder, "*.s2p")))
    f0, _ = read_s2p(files[0]); rows=[]; g0=[]
    for fp in files:
        fn = os.path.basename(fp)
        if fn.lower().startswith("air"): continue
        c = conc(fn)
        if c is None: continue
        f, s = read_s2p(fp)
        if len(s)!=len(f0): continue
        rows.append((c, s))
        if c == 0: g0.append(s)
    spectra[sess] = rows; base[sess] = np.mean(g0, axis=0)

# detect notch frequencies from combined baseline (local minima < -10 dB, 1-6 GHz)
bcomb = (base[1] + base[2]) / 2
band = (f0 >= 1e9) & (f0 <= 6e9)
notch_idx = [i for i in range(1, len(f0)-1)
             if band[i] and bcomb[i] < bcomb[i-1] and bcomb[i] < bcomb[i+1] and bcomb[i] < -10]
notch_f = [round(f0[i]/1e9, 2) for i in notch_idx]
i43 = notch_idx[int(np.argmin([abs(ff-4.3) for ff in notch_f]))]   # the 4.3 GHz notch index
print("Detected notches (GHz):", notch_f)

# feature matrix: baseline-corrected dS11 at each notch, per sample
X_all=[]; y=[]; sess_arr=[]
for sess, rows in spectra.items():
    for c, s in rows:
        lv = level(c)
        if lv is None: continue
        d = s - base[sess]
        X_all.append([d[i] for i in notch_idx]); y.append(lv); sess_arr.append(sess)
X_all = np.array(X_all); y = np.array(y); sess_arr = np.array(sess_arr)
col43 = notch_idx.index(i43)
print("samples:", len(y), "| notch features:", X_all.shape[1])

def loso(X):
    a=[]
    for te in [1,2]:
        tr=sess_arr!=te; ts=sess_arr==te
        m=make_pipeline(StandardScaler(),LDA()).fit(X[tr],y[tr]); a.append(accuracy_score(y[ts],m.predict(X[ts])))
    return np.mean(a)
def cv(X):
    a=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=1).split(X,y):
        m=make_pipeline(StandardScaler(),LDA()).fit(X[tr],y[tr]); a.append(accuracy_score(y[te],m.predict(X[te])))
    return np.mean(a)

# rank notches by individual LOSO power (on TRAIN-session only would be ideal; here just to report)
single = sorted(range(X_all.shape[1]), key=lambda j: -loso(X_all[:,[j]]))
print("\nfeature-set                         (B)CV    (C)LOSO")
sets = {
 "single 4.3 GHz notch":            [col43],
 "top-2 notches":                   single[:2],
 "top-3 notches":                   single[:3],
 "top-5 notches":                   single[:5],
 "ALL %d notches" % X_all.shape[1]: list(range(X_all.shape[1])),
}
for name, cols in sets.items():
    print(f"{name:34s}{cv(X_all[:,cols])*100:7.1f}%{loso(X_all[:,cols])*100:9.1f}%")

# honest forward selection: choose subset using ONLY the training session, eval on test session
def loso_fwd(maxk=6):
    accs=[]
    for te in [1,2]:
        tr=sess_arr!=te; ts=sess_arr==te
        Xtr,ytr,Xts,yts=X_all[tr],y[tr],X_all[ts],y[ts]
        chosen=[]; remaining=list(range(X_all.shape[1]))
        # internal CV on train to pick features (no peek at test)
        best=chosen
        while len(chosen)<maxk and remaining:
            scored=[]
            for j in remaining:
                cols=chosen+[j]; sc=[]
                for itr,ite in StratifiedKFold(3,shuffle=True,random_state=2).split(Xtr,ytr):
                    m=make_pipeline(StandardScaler(),LDA()).fit(Xtr[itr][:,cols],ytr[itr]); sc.append(accuracy_score(ytr[ite],m.predict(Xtr[ite][:,cols])))
                scored.append((np.mean(sc),j))
            scored.sort(reverse=True); chosen.append(scored[0][1]); remaining.remove(scored[0][1])
        m=make_pipeline(StandardScaler(),LDA()).fit(Xtr[:,chosen],ytr); accs.append(accuracy_score(yts,m.predict(Xts[:,chosen])))
    return np.mean(accs)
print(f"\n{'forward-selected (nested, honest)':34s}{'':7s}{loso_fwd()*100:9.1f}%  <- subset chosen on train only")
