"""
Physically-grounded feature extraction from raw .s2p + honest LOSO validation.
Run from: Glucose_Data_extracted/Glucose_Data/Raw data from vna/

Pipeline:
  1. Parse Touchstone .s2p (Hz, Real/Imag) -> S11(dB) = 20log10|S11|, 101 pts, 1-6 GHz
  2. Per-session baseline = mean S11(dB) of the Glu0 (PBS, 0 mg/dL) reference scans
  3. Features: notch depth at the ~4.3 GHz resonance (absolute and baseline-corrected dS11)
  4. Leave-One-Session-Out (LOSO) classification (4 glucose levels) and regression (mg/dL)
Key finding: single physical notch feature generalizes across sessions (~76%),
whereas the full 101-pt spectrum overfits to session (97% within, 43% across),
and the original 3 entropy features collapse to chance across sessions.
"""
import numpy as np, glob, re, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

SESSIONS = {1: "032067 เก็บข้อมูล Glucose Blood 1", 2: "032167 เก็บข้อมูล Glucose Blood 2"}

def read_s2p(path):
    fr, s11, fmt, unit = [], [], 'RI', 'Hz'
    for line in open(path, encoding='latin-1'):
        s = line.strip()
        if not s or s.startswith('!'):
            continue
        if s.startswith('#'):
            t = s.lower().split()
            unit = {'ghz':'GHz','mhz':'MHz','khz':'kHz','hz':'Hz'}.get(next((x for x in t if x in ('ghz','mhz','khz','hz')), 'hz'))
            fmt = 'MA' if 'ma' in t else 'RI'
            continue
        p = s.split()
        if len(p) < 3:
            continue
        try:
            v = [float(x) for x in p]
        except ValueError:
            continue
        fr.append(v[0])
        s11.append(complex(v[1], v[2]) if fmt == 'RI' else v[1]*np.exp(1j*np.deg2rad(v[2])))
    mult = {'Hz':1,'kHz':1e3,'MHz':1e6,'GHz':1e9}[unit]
    return np.array(fr)*mult, 20*np.log10(np.abs(np.array(s11)) + 1e-15)

conc = lambda fn: (int(re.search(r'Glu(\d+)', fn).group(1)) if re.search(r'Glu(\d+)', fn) else None)
def level(c):
    if 45 <= c <= 65:  return 0
    if 70 <= c <= 110: return 1
    if 125 <= c <= 200: return 2
    if 220 <= c <= 300: return 3
    return None

def build():
    SP, META = [], []
    for sess, folder in SESSIONS.items():
        files = sorted(glob.glob(os.path.join(folder, "*.s2p")))
        f0, _ = read_s2p(files[0])
        win = (f0 >= 4.0e9) & (f0 <= 4.6e9); widx = np.where(win)[0]
        base = np.mean([read_s2p(fp)[1] for fp in files if conc(os.path.basename(fp)) == 0], axis=0)
        bidx = widx[np.argmin(base[widx])]
        for fp in files:
            fn = os.path.basename(fp)
            if fn.lower().startswith('air'):
                continue
            c = conc(fn)
            if c is None:
                continue
            f, s = read_s2p(fp)
            if len(s) != len(f0):
                continue
            SP.append(s - base)
            META.append(dict(sess=sess, conc=c, lvl=level(c),
                             notch_abs=s[bidx], dnotch=s[bidx] - base[bidx], fres=f0[bidx]/1e9))
    return np.array(SP), pd.DataFrame(META)

if __name__ == '__main__':
    SP, df = build()
    df.to_csv(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 's2p_notch_features.csv'), index=False)
    sess = df.sess.values; lvl = df.lvl.fillna(-1).astype(int).values; cc = df.conc.values
    print("samples:", len(df), "| spectra:", SP.shape)

    print("\nLOSO CLASSIFICATION (4 levels, chance 25%)")
    for name, cols in [("notch_abs", ['notch_abs']), ("dnotch", ['dnotch'])]:
        accs = []
        for te in [1, 2]:
            m = (lvl >= 0)
            tr = (sess != te) & m; ts = (sess == te) & m
            sc = StandardScaler().fit(df.loc[tr, cols]); md = LogisticRegression(max_iter=5000).fit(sc.transform(df.loc[tr, cols]), lvl[tr])
            accs.append(accuracy_score(lvl[ts], md.predict(sc.transform(df.loc[ts, cols]))))
        print(f"  {name:12s} mean={np.mean(accs)*100:.1f}%  folds={[round(a*100,1) for a in accs]}")
    # full spectrum (overfits to session)
    accs = []
    for te in [1, 2]:
        m = lvl >= 0; tr = (sess != te) & m; ts = (sess == te) & m
        md = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(SP[tr], lvl[tr])
        accs.append(accuracy_score(lvl[ts], md.predict(SP[ts])))
    print(f"  {'full dS11':12s} mean={np.mean(accs)*100:.1f}%  folds={[round(a*100,1) for a in accs]}  (overfits session)")

    print("\nLOSO REGRESSION (predict mg/dL)")
    for name, mk in [("notch (RF)", lambda: RandomForestRegressor(300, random_state=0)),
                     ("full dS11 Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)))]:
        maes, r2s = [], []
        for te in [1, 2]:
            tr = sess != te; ts = sess == te
            Xtr = SP[tr] if 'full' in name else df.loc[tr, ['notch_abs']]
            Xts = SP[ts] if 'full' in name else df.loc[ts, ['notch_abs']]
            md = mk().fit(Xtr, cc[tr]); p = md.predict(Xts)
            maes.append(mean_absolute_error(cc[ts], p)); r2s.append(r2_score(cc[ts], p))
        print(f"  {name:16s} MAE={np.mean(maes):.1f} mg/dL  R2={np.mean(r2s):.3f}")
