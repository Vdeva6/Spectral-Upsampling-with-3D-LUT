import numpy as np
import matplotlib.pyplot as plt
import os
from parse_colorchecker import parse_colorchecker
from parse_coeff_table import load_lut_from_coeff

# Utility functions
def load_cmf(path="data/cie-cmf.txt"):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]

def load_illuminant(path="data/D65.5nm"):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]

def cie_xyz_from_spectrum(reflectance, cmf, illuminant):
    k = 100 / np.sum(cmf[:, 1] * illuminant)
    reflectance_weighted = reflectance * illuminant  # shape (24, 36)
    return k * np.dot(reflectance_weighted, cmf)


def xyz_to_rgb(XYZ):
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    return np.clip(np.dot(XYZ, M.T), 0, 1)

# Spectral model from spectral-model.py
def S(x):
    return 0.5 + (x / (2 * np.sqrt(1 + x**2)))

def spectrum(wavelengths, c0, c1, c2):
    x = (c0 * wavelengths**2) + (c1 * wavelengths) + c2
    return S(x)

# Trilinear interpolation lookup
def lerp(a, b, f):
    return (1 - f) * a + f * b

def srgb_gamma_encode(rgb):
    """Apply sRGB gamma encoding to linear RGB."""
    a = 0.055
    encoded = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        (1 + a) * np.power(rgb, 1/2.4) - a
    )
    return np.clip(encoded, 0.0, 1.0)


def look_up_coefficients(rgb, lut):
    r, g, b = rgb
    scale = lut.shape[0] - 1
    r_idx = r * scale
    g_idx = g * scale
    b_idx = b * scale

    i0, j0, k0 = int(np.floor(r_idx)), int(np.floor(g_idx)), int(np.floor(b_idx))
    i1, j1, k1 = min(i0 + 1, scale), min(j0 + 1, scale), min(k0 + 1, scale)
    fr, fg, fb = r_idx - i0, g_idx - j0, b_idx - k0

    V000 = lut[i0, j0, k0]
    V100 = lut[i1, j0, k0]
    V010 = lut[i0, j1, k0]
    V110 = lut[i1, j1, k0]
    V001 = lut[i0, j0, k1]
    V101 = lut[i1, j0, k1]
    V011 = lut[i0, j1, k1]
    V111 = lut[i1, j1, k1]

    R1 = lerp(V000, V100, fr)
    R2 = lerp(V010, V110, fr)
    R3 = lerp(V001, V101, fr)
    R4 = lerp(V011, V111, fr)
    G1 = lerp(R1, R2, fg)
    G2 = lerp(R3, R4, fg)
    return lerp(G1, G2, fb)

# Main Logic
reflectance, wavelengths = parse_colorchecker()

wl_cmf, cmf = load_cmf()
wl_D65, illuminant = load_illuminant()

cmf_interp = np.vstack([np.interp(wavelengths, wl_cmf, cmf[:, i]) for i in range(3)]).T
D65_interp = np.interp(wavelengths, wl_D65, illuminant)

XYZ = cie_xyz_from_spectrum(reflectance, cmf_interp, D65_interp)
RGB = xyz_to_rgb(XYZ)

lut = load_lut_from_coeff("data/generated_lut.npy")

fig, ax = plt.subplots()
for i in range(reflectance.shape[0]):
    rgb = RGB[i]
    c0, c1, c2 = look_up_coefficients(rgb, lut)
    recon = spectrum(wavelengths, c0, c1, c2)
    ax.plot(wavelengths, reflectance[i], label=f'GT {i+1}', linestyle='--')
    ax.plot(wavelengths, recon, label=f'Pred {i+1}', alpha=0.75)

ax.set_title("Reflectance: Ground Truth vs Predicted (Parametric LUT Model)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.legend(ncol=2, fontsize=6)
plt.tight_layout()
plt.show()
