# -*- coding: utf-8 -*-
"""Regenerate blood S11 spectrum (Fig 8) and calibration (Fig 9) directly from raw .s2p.
   Format: '# Hz S RI R 50' -> cols: freq, S11re, S11im, ... ; |S11|dB = 20log10(hypot(re,im)).
   Saves to figures/fig_blood_spectrum.png and figures/fig_blood_calibration.png (new names; originals untouched)."""
import os, re, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "Glucose_Data_extracted/Glucose_Data/Raw data from vna"
S1 = os.path.join(BASE, "032067 เก็บข้อมูล Glucose Blood 1")
S2 = os.path.join(BASE, "032167 เก็บข้อมูล Glucose Blood 2")

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
            fr = float(p[0]); a = float(p[1]); b = float(p[2])
        except ValueError:
            continue
        f.append(fr); re_.append(a); im_.append(b)
    f = np.array(f); mag = np.hypot(np.array(re_), np.array(im_))
    fghz = f / 1e9 if f.max() > 1e6 else f
    return fghz, 20 * np.log10(mag)

def conc_files(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.s2p")):
        b = os.path.basename(p)
        if "Air" in b:
            continue
        m = re.search(r"Glu(\d+)", b)
        if not m:
            continue
        c = int(m.group(1))
        out.setdefault(c, []).append(p)
    return out

def mean_spectrum(paths):
    specs = []
    fref = None
    for p in paths:
        fg, db = s11_db(p)
        if len(db) != 101:
            continue
        fref = fg
        specs.append(db)
    return fref, np.mean(specs, axis=0)

cf1 = conc_files(S1)
cf2 = conc_files(S2)
NOTCH = 4.30
SHOW = [0, 45, 50, 55, 70, 120, 200, 300]

# ---------- Figure 8: spectrum + zoom inset ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(SHOW)))
fref = None
for c, col in zip(SHOW, cmap):
    if c not in cf1:
        continue
    fg, db = mean_spectrum(cf1[c])
    fref = fg
    lbl = "0 mg/dL (PBS)" if c == 0 else f"{c} mg/dL"
    axL.plot(fg, db, lw=1.0, color=col, label=lbl)
    axR.plot(fg, db, lw=1.4, color=col, label=lbl)
axL.set_xlim(1, 6); axL.set_xlabel("Frequency (GHz)"); axL.set_ylabel(r"$S_{11}$ (dB)")
axL.axvspan(4.2, 4.4, color="0.85", alpha=0.6)
axL.set_title("(a) Synthetic blood, 1–6 GHz", fontsize=10)
axR.set_xlim(4.255, 4.345); axR.set_ylim(-16.55, -15.55)
axR.set_xlabel("Frequency (GHz)"); axR.set_ylabel(r"$S_{11}$ (dB)")
axR.set_title("(b) Resonant notch near 4.3 GHz (zoom)", fontsize=10)
axR.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
axR.annotate("0 mg/dL", xy=(4.30, -16.03), xytext=(4.262, -15.75), fontsize=7.5,
             ha="left", arrowprops=dict(arrowstyle="->", color="#3a3a8a"))
axR.annotate("300 mg/dL\n(notch deepens)", xy=(4.30, -16.35), xytext=(4.305, -16.5),
             fontsize=7.5, ha="left", arrowprops=dict(arrowstyle="->", color="#8a3a3a"))
for ax in (axL, axR):
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/fig_blood_spectrum.png", dpi=170, bbox_inches="tight")
plt.close()

# ---------- Figure 9: calibration (notch@4.3 vs conc, both sessions) ----------
def calib(cf):
    xs, ys = [], []
    for c in sorted(cf):
        fg, db = mean_spectrum(cf[c])
        j = int(np.argmin(np.abs(fg - NOTCH)))
        xs.append(c); ys.append(db[j])
    return np.array(xs), np.array(ys)

x1, y1 = calib(cf1)
x2, y2 = calib(cf2)
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(x1, y1, "o", ms=4, color="#2c6fbb", label="Session 1")
ax.plot(x2, y2, "s", ms=4, color="#c0392b", label="Session 2")
xall = np.concatenate([x1, x2]); yall = np.concatenate([y1, y2])
m, b = np.polyfit(xall, yall, 1)
ss = 1 - np.sum((yall - (m * xall + b)) ** 2) / np.sum((yall - yall.mean()) ** 2)
xx = np.linspace(0, 300, 50)
ax.plot(xx, m * xx + b, "k--", lw=1, label=f"linear fit (R²={ss:.2f})")
ax.set_xlabel("Glucose concentration (mg/dL)")
ax.set_ylabel(r"Notch depth $S_{11}$ at 4.3 GHz (dB)")
ax.invert_yaxis()  # deeper (more negative) downward
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
ax.set_title(f"Notch deepens by ~{abs(y1[-1]-y1[0]):.2f} dB over 0–300 mg/dL", fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig_blood_calibration.png", dpi=170, bbox_inches="tight")
plt.close()

print("session1 notch@4.3:", {int(c): round(float(v), 3) for c, v in zip(x1, y1) if c in SHOW})
print("slope dB/(mg/dL):", round(m, 5), "R2:", round(ss, 3))
print("saved figures/fig_blood_spectrum.png and figures/fig_blood_calibration.png")
