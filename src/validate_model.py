import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from data.cie_xyz_rgb import reflectance_to_rgb
from src.spectral_model import look_up_coefficients, spectrum
from src.parse_colorchecker import parse_colorchecker

# Load the ColorChecker reflectance data
from src.parse_colorchecker import parse_colorchecker
reflectances, wavelengths = parse_colorchecker('data/ColorChecker_spectra.txt')

# Load LUT table
lut = np.load('data/generated_lut.npy')

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

errors = []

# Validate across all patches
for i, real_reflectance in enumerate(reflectances):
    # Step 1: reflectance → RGB
    rgb = reflectance_to_rgb(real_reflectance)

    # Step 2: RGB → coefficients
    c0, c1, c2 = look_up_coefficients(rgb)

    # Step 3: coefficients → predicted spectrum
    predicted_refl = spectrum(wavelengths, c0, c1, c2)

    # Step 4: compute RMSE
    err = compute_rmse(predicted_refl, real_reflectance)
    errors.append(err)

    # Optional: plot first 3 results
    if i < 3:
        plt.plot(wavelengths, real_reflectance, label="Measured", color='black')
        plt.plot(wavelengths, predicted_refl, label="Predicted", linestyle='--', color='red')
        plt.title(f"Patch {i + 1}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.show()

# Summary
errors = np.array(errors)
print(f"Average RMSE over {len(errors)} patches: {errors.mean():.4f}")
print(f"Min RMSE: {errors.min():.4f}, Max RMSE: {errors.max():.4f}")
