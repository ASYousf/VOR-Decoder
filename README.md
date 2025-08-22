# VOR Decoder (Python)

This project contains a Python implementation of a **VHF Omnidirectional Range (VOR) decoder**.  
It processes recorded IQ samples (`.wav`) to extract the 30 Hz reference and variable tones, estimate their phase difference, and compute the **bearing angle**.

---

## 🚀 Features
- Loads IQ samples from a `.wav` recording.
- Mixes and filters signals down to audio baseband.
- Extracts the **30 Hz variable (AM)** and **30 Hz reference (FM)** tones.
- Computes phase difference to estimate the **VOR bearing**.
- Generates plots:
  - Time-domain zoom of 30 Hz signals
  - Instantaneous phase difference trace
  - Histogram of phase differences

---

## 📂 File
- **`VOR_1_v25_ML.py`** → Main Python script for decoding.

---

## 🛠 Requirements
Install dependencies before running:
```bash
pip install numpy matplotlib scipy scikit-learn
