import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from src.parse_colorchecker import load_colorchecker_spectra
from data.cie_xyz_rgb import reflectance_to_rgb
from src.spectral_model import spectrum

# Wavelengths
orig_waves = np.arange(380, 740, 15)      # Original data shape (24 bands)
target_waves = np.arange(380, 781, 5)     # 81 bands used in basis, D65, CMFs

# === Helper: Upsample reflectance to 5nm spacing
def upsample_reflectance(reflectance, orig_waves, target_waves):
    interp = interp1d(orig_waves, reflectance, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interp(target_waves)

# === Helper: Minimize MSE to get best-fit c0, c1, c2
def fit_coefficients(reflectance_target, wavelengths):
    def loss_fn(c):
        pred = spectrum(wavelengths, c[0], c[1], c[2])
        return np.mean((pred - reflectance_target) ** 2)

    result = minimize(loss_fn, x0=np.zeros(3), method='L-BFGS-B')
    return result.x

# === Main: Load spectra and generate LUT
print("Loading ColorChecker spectra...")
spectra, _ = load_colorchecker_spectra("data/ColorChecker_spectra.txt")

lut = []  # Each entry: (r, g, b, c0, c1, c2)

for i, reflectance in enumerate(spectra):
    reflectance_upsampled = upsample_reflectance(reflectance, orig_waves, target_waves)
    rgb = reflectance_to_rgb(reflectance_upsampled)
    c0, c1, c2 = fit_coefficients(reflectance_upsampled, target_waves)
    lut.append((*rgb, c0, c1, c2))
    print(f"Sample {i}: RGB = {rgb}, Coeffs = {c0:.3f}, {c1:.3f}, {c2:.3f}")

lut_array = np.array(lut, dtype=np.float32)
np.save("data/lut_from_colorchecker.npy", lut_array)
print("\n✅ LUT saved to data/lut_from_colorchecker.npy")
