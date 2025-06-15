import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from src.parse_colorchecker import load_colorchecker_spectra
from data.cie_xyz_rgb import reflectance_to_rgb
from src.spectral_model import spectrum, look_up_coefficients, load_lut_from_file

# Load ColorChecker reflectance spectra
spectra, orig_waves = load_colorchecker_spectra('data/ColorChecker_spectra.txt')

# Wavelengths used for prediction and CMFs
target_waves = np.arange(380, 781, 5)

# Load the LUT you just built
lut = load_lut_from_file('data/lut_from_colorchecker.npy')

# Helper to upsample 24-band data to 5nm spacing
def upsample_reflectance(reflectance, orig_waves, target_waves):
    interp = interp1d(orig_waves, reflectance, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interp(target_waves)

# Plot predictions vs. ground truth
for i, reflectance in enumerate(spectra):
    reflectance_upsampled = upsample_reflectance(reflectance, orig_waves, target_waves)
    rgb = reflectance_to_rgb(reflectance_upsampled)
    c0, c1, c2 = look_up_coefficients(rgb, lut)
    predicted = spectrum(target_waves, c0, c1, c2)
    
    mse = np.mean((predicted - reflectance_upsampled) ** 2)

    plt.figure()
    plt.plot(target_waves, reflectance_upsampled, label='Ground Truth')
    plt.plot(target_waves, predicted, label='Predicted')
    plt.title(f'Sample {i+1} — MSE: {mse:.6f}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.show()
