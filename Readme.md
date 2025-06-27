# Direction-of-Arrival Estimation for Wideband Chirps in the Fractional Fourier Domain

## 📖 Overview

This repository presents algorithms for **direction-of-arrival (DoA) estimation** of wideband linear chirp signals using techniques in the **fractional Fourier domain**, particularly focusing on multi-target scenarios and uniform linear arrays (ULAs).

The core contribution includes the use of the **discrete simplified fractional Fourier transform (DSmFrFT)**, along with a series of techniques for improving DoA estimation performance and reducing computational complexity.

---

## 🧠 Key Features

- **DoA estimation for wideband chirp signals** using fractional Fourier analysis.
- Implementation of **subspace-based algorithms** with spatial smoothing and forward-backward averaging:
  - FBSS-MUSIC
  - FBSS-ESPRIT
- **Peak alignment preprocessing** to improve subspace-based algorithm accuracy.
- **Multi-line fitting** methods for multi-target cases:
  - Piecewise linear regression
  - Line detection via **Hough transform**
- **Python 3.9.7** implementation in **Jupyter Notebooks** with numerical simulations and visualizations.

---

## 📁 Repository Structure

```
├── single_target_simulation.ipynb
├── multi_target_simulation.ipynb
├── src/
│   ├── Chirp.py
│   ├── frft_utils.py
│   └── doa_utils.py
├── README.md
└── requirements.txt
```

---

## 🛠 Requirements

- Python 3.9.7
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- OpenCV (for Hough Transform)
- Jupyter

Install using:

```bash
pip install -r requirements.txt
```

---

## 📊 Example Output

Example results are shown directly in the output cells of the Jupyter notebooks `single_target_simulation.ipynb` and `multi_target_simulation.ipynb`,  corresponding to the single- and multi-target scenarios, respectively. For the latter, the following aspects are addressed:
```
- Signal Modeling
- Preprocessing
- Multi-target direction-of-arrival (DoA) estimation
  ├──  Piecewise linear regression
  ├──  Line detection in Hough space
  ├──  MUSIC with spatial smoothing
  └──  ESPRIT with spatial smoothing
```
---

## 📄 Reference

If you use this code or build upon this work, please cite the following:

> **E. G. Huampo, J. B. Lima and J. R. d. O. Neto,** "Direction-of-Arrival Estimation for Wideband Chirps via Multi-Line Fitting in the Fractional Fourier Domain," in *IEEE Access*, vol. 13, pp. 107328-107342, 2025, doi: 10.1109/ACCESS.2025.3581843.

---

## 📬 Contact

For questions, suggestions, or collaborations, feel free to reach out via GitHub Issues or email.

---

## 📜 License

This project is licensed under the MIT License.
