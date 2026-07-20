==============================================================================
 SUPPLEMENTARY DATA AND CODE
==============================================================================

Raw measurements and Python scripts to reproduce the Leave-One-Session-Out
(LOSO) validation, the confusion matrix, and the leakage demonstrations in the
manuscript:

  "Cross-Session Glucose Level Estimation with a CSRR-Loaded Microwave Sensor:
   A Baseline-Corrected Resonant Feature Outperforms Full-Spectrum Deep Learning"


------------------------------------------------------------------------------
 CONTENTS
------------------------------------------------------------------------------

  code/                       Python scripts + s2p_notch_features.csv (features)
  raw_data/session1_blood/    78 .s2p files  -  measurement Session 1
  raw_data/session2_blood/    68 .s2p files  -  measurement Session 2


------------------------------------------------------------------------------
 RAW .s2p FORMAT
------------------------------------------------------------------------------

Each .s2p file is one droplet measurement: 101 points over 1-6 GHz, in
"# Hz S RI R 50" format (S11 given as real / imaginary parts).

Filenames encode the glucose concentration, e.g.
  "Blood Glu120 1 ... .s2p"  =  120 mg/dL, repeat 1
  "Blood Glu0 PBS ... .s2p"   =  per-session 0 mg/dL (PBS) reference

Convert to magnitude in dB:
  |S11| (dB) = 20 * log10( hypot(real, imag) )

Glucose level mapping (0 and 120 mg/dL are excluded):
  Level 1 = 45-65       Level 2 = 70-110
  Level 3 = 125-200     Level 4 = 220-300


------------------------------------------------------------------------------
 DATASET COMPOSITION OF THE LOSO CONFUSION MATRIX (Figure 12)
------------------------------------------------------------------------------

  126 measurements total.

      Level:        1     2     3     4    Total
      Session 1:   21    15    15    15     66
      Session 2:   15    15    15    15     60

  Leave-One-Session-Out:
      Session 1 held out:  55 / 66  = 83.3 %
      Session 2 held out:  49 / 60  = 81.7 %
      Overall:            104 / 126 = 82.5 %


------------------------------------------------------------------------------
 HOW TO REPRODUCE
------------------------------------------------------------------------------

Requirements: Python 3 with numpy, scipy, scikit-learn, matplotlib
              (TensorFlow is also required for the 1D-CNN).

In each script, set the two session-folder paths near the top of the file to:
      raw_data/session1_blood   and   raw_data/session2_blood

Then run, for example:

  python code/multi_notch_experiment.py
        -> single 4.3 GHz notch, LOSO accuracy = 82.5 %
           (also the multi-notch leakage: top-5 = 88.8% leaky -> 38% honest)

  python code/cnn_experiment.py
        -> full 101-point spectrum with LDA / MLP / 1D-CNN,
           augment-before-split (leaky) vs Leave-One-Session-Out (~98% -> 43-44%)

  python code/correct_split_experiment.py
        -> augment-before-split leakage (92.5% of test rows are duplicates)

  python code/gen_blood_figs_from_raw.py
        -> blood S11 spectrum and calibration figures, recomputed from raw .s2p


------------------------------------------------------------------------------
 NOTES
------------------------------------------------------------------------------

- The single 4.3 GHz notch is pre-specified on physical grounds, not selected
  from the data. Scripts that *select* notches (multi-notch) demonstrate why
  post-hoc feature selection leaks across sessions.
- Random seeds are fixed where results depend on initialisation; the 1D-CNN
  result is reported as a mean over three seeds (44% +/- 5).
==============================================================================
