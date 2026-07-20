"""
Leakage-free analysis package for the revised paper.
Run from: Glucose_Data_extracted/Glucose_Data/Raw data from vna/
Produces (saved to ../../../figures/):
  fig_calibration.png     - dNotch vs glucose, both sessions overlaid + linear fit/R2
  fig_confusion_loso.png  - LOSO confusion matrix (per-measurement)
  fig_clarke_grid.png     - Clarke Error Grid for LOSO regression
and prints classification report + regression metrics + Clarke zone %.
Headline honest result: single notch-depth feature, LDA, Leave-One-Session-Out:
  per-measurement 4-level accuracy ~82.5%, rep-averaged ~95%, regression R2~0.67.
"""
import numpy as np, glob, re, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score

OUT = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "figures")); os.makedirs(OUT, exist_ok=True)
SESS = {1: "032067 เก็บข้อมูล Glucose Blood 1", 2: "032167 เก็บข้อมูล Glucose Blood 2"}

def read_s2p(p):
    fr, s11, fmt, unit = [], [], 'RI', 'Hz'
    for line in open(p, encoding='latin-1'):
        s = line.strip()
        if not s or s.startswith('!'): continue
        if s.startswith('#'):
            t = s.lower().split(); unit = {'ghz':'GHz','mhz':'MHz','khz':'kHz','hz':'Hz'}.get(next((x for x in t if x in ('ghz','mhz','khz','hz')), 'hz')); fmt = 'MA' if 'ma' in t else 'RI'; continue
        q = s.split()
        if len(q) < 3: continue
        try: v = [float(x) for x in q]
        except ValueError: continue
        fr.append(v[0]); s11.append(complex(v[1], v[2]) if fmt == 'RI' else v[1]*np.exp(1j*np.deg2rad(v[2])))
    m = {'Hz':1,'kHz':1e3,'MHz':1e6,'GHz':1e9}[unit]
    return np.array(fr)*m, 20*np.log10(np.abs(np.array(s11)) + 1e-15)

conc = lambda fn: (int(re.search(r'Glu(\d+)', fn).group(1)) if re.search(r'Glu(\d+)', fn) else None)
def level(c):
    if 45 <= c <= 65: return 0
    if 70 <= c <= 110: return 1
    if 125 <= c <= 200: return 2
    if 220 <= c <= 300: return 3
    return None
NAMES = ['Lv.1 (45-65)', 'Lv.2 (70-110)', 'Lv.3 (125-200)', 'Lv.4 (220-300)']

def build():
    rows = []
    for sess, folder in SESS.items():
        files = sorted(glob.glob(os.path.join(folder, "*.s2p")))
        f0, _ = read_s2p(files[0]); widx = np.where((f0 >= 4.0e9) & (f0 <= 4.6e9))[0]
        base = np.mean([read_s2p(fp)[1] for fp in files if conc(os.path.basename(fp)) == 0], axis=0)
        bidx = widx[np.argmin(base[widx])]
        for fp in files:
            fn = os.path.basename(fp)
            if fn.lower().startswith('air'): continue
            c = conc(fn)
            if c is None: continue
            f, s = read_s2p(fp)
            if len(s) != len(f0): continue
            rows.append(dict(sess=sess, conc=c, lvl=level(c), notch_abs=s[bidx], dnotch=s[bidx]-base[bidx], fres=f0[bidx]/1e9))
    return pd.DataFrame(rows)

df = build()

# ---------- LOSO classification (per-measurement), pooled predictions ----------
def loso_pred_clf(data, cols):
    yt, yp, ss = [], [], []
    for te in [1, 2]:
        tr = data[data.sess != te]; ts = data[data.sess == te]
        m = make_pipeline(StandardScaler(), LDA()).fit(tr[cols], tr.lvl)
        yt += list(ts.lvl); yp += list(m.predict(ts[cols])); ss += list(ts.sess)
    return np.array(yt, int), np.array(yp, int), np.array(ss)

lev = df.dropna(subset=['lvl']).copy(); lev['lvl'] = lev['lvl'].astype(int)
yt, yp, _ = loso_pred_clf(lev, ['notch_abs'])
print("="*64, "\nLOSO 4-LEVEL CLASSIFICATION  (single feature notch_abs, LDA)\n", "="*64)
print(f"Per-measurement overall accuracy = {accuracy_score(yt, yp)*100:.1f}%")
print(classification_report(yt, yp, target_names=NAMES, digits=4))
# rep-averaged
ra = lev.groupby(['sess','conc'], as_index=False).mean(numeric_only=True); ra['lvl'] = ra['conc'].apply(level).astype(int)
yt2, yp2, _ = loso_pred_clf(ra, ['notch_abs'])
print(f"Rep-averaged overall accuracy = {accuracy_score(yt2, yp2)*100:.1f}%  (n={len(yt2)})")

# ---------- LOSO regression, pooled predictions ----------
def loso_pred_reg(data, cols):
    rt, rp = [], []
    for te in [1, 2]:
        tr = data[data.sess != te]; ts = data[data.sess == te]
        m = LinearRegression().fit(tr[cols], tr.conc)
        rt += list(ts.conc); rp += list(m.predict(ts[cols]))
    return np.array(rt, float), np.array(rp, float)
rt, rp = loso_pred_reg(df, ['notch_abs'])
print("\n", "="*64, "\nLOSO REGRESSION (predict mg/dL)\n", "="*64)
print(f"MAE = {mean_absolute_error(rt, rp):.1f} mg/dL   R2 = {r2_score(rt, rp):.3f}")

# ---------- Clarke Error Grid zones ----------
def clarke(ref, pred):
    z = []
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or abs(p - r) <= 0.2*r: z.append('A')
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180): z.append('E')
        elif ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7/5)*r - 182): z.append('C')
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175/3 and 70 <= p <= 180) or ((175/3 <= r <= 70) and p >= 1.2*r): z.append('D')
        else: z.append('B')
    return np.array(z)
zones = clarke(rt, rp); zc = {z: int(np.sum(zones == z)) for z in 'ABCDE'}
print("Clarke zones:", {k: f'{v} ({v/len(zones)*100:.0f}%)' for k, v in zc.items()}, f"| A+B = {(zc['A']+zc['B'])/len(zones)*100:.0f}%")

# ---------- FIGURES ----------
# 1. calibration curve
plt.figure(figsize=(6,4.2))
for sess, mk, cl in [(1,'o','#1f77b4'), (2,'s','#d62728')]:
    g = df[df.sess==sess].groupby('conc')['dnotch'].agg(['mean','std']).reset_index()
    plt.errorbar(g.conc, g['mean'], yerr=g['std'], fmt=mk, color=cl, capsize=2, ms=4, label=f'Session {sess}')
a, b = np.polyfit(df.conc, df.dnotch, 1); xx = np.linspace(0,300,50)
r2 = r2_score(df.dnotch, a*df.conc+b)
plt.plot(xx, a*xx+b, 'k--', lw=1, label=f'fit: y={a:.5f}x{b:+.3f}\n$R^2$={r2:.3f}')
plt.xlabel('Glucose concentration (mg/dL)'); plt.ylabel(r'$\Delta S_{11}$ notch depth @4.3GHz (dB)')
plt.title('Baseline-corrected notch response (two independent sessions)')
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f'{OUT}/fig_calibration.png', dpi=150); plt.close()

# 2. confusion matrix
cm = confusion_matrix(yt, yp)
plt.figure(figsize=(5,4.3)); plt.imshow(cm, cmap='Blues')
for i in range(4):
    for j in range(4): plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.xticks(range(4), [n.split()[0] for n in NAMES]); plt.yticks(range(4), [n.split()[0] for n in NAMES])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'LOSO Confusion Matrix (acc={accuracy_score(yt,yp)*100:.1f}%)')
plt.colorbar(); plt.tight_layout(); plt.savefig(f'{OUT}/fig_confusion_loso.png', dpi=150); plt.close()

# 3. Clarke grid
plt.figure(figsize=(5.2,5))
plt.scatter(rt, rp, s=18, c='#2c7fb8', edgecolor='k', lw=.3, alpha=.8)
mx=350; plt.plot([0,mx],[0,mx],'k-',lw=.6)
plt.plot([0,mx],[0,mx*1.2],'k:',lw=.5); plt.plot([0,mx],[0,mx*.8],'k:',lw=.5)
plt.xlim(0,mx); plt.ylim(0,mx); plt.xlabel('Reference glucose (mg/dL)'); plt.ylabel('Predicted glucose (mg/dL)')
plt.title(f"Clarke Error Grid (LOSO)  A+B = {(zc['A']+zc['B'])/len(zones)*100:.0f}%")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f'{OUT}/fig_clarke_grid.png', dpi=150); plt.close()

print(f"\nFigures saved to: {OUT}")
print(" - fig_calibration.png, fig_confusion_loso.png, fig_clarke_grid.png")
