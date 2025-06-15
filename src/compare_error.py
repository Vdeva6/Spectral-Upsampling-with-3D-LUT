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
    illuminant = illuminant / (np.sum(cmf[:, 1] * illuminant) / 100)
    k = 100 / np.sum(cmf[:, 1] * illuminant)
    return k * np.dot(reflectance * illuminant[np.newaxis, :], cmf)

def xyz_to_rgb(XYZ):
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    return np.clip(np.dot(XYZ, M.T), 0, 1)

def lerp(a, b, f):
    return (1 - f) * a + f * b

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

def S(x):
    return 0.5 + 0.5 * x * (1 / np.sqrt(1 + x**2))  # Prevents overflow

def analytic_spectrum(wavelengths, c0, c1, c2):
    return S(c0 * wavelengths**2 + c1 * wavelengths + c2)

def to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1/2.4) - 0.055)

# Load data
reflectance, ref_wl = parse_colorchecker()
cmf_wl, cmf = load_cmf()
ill_wl, illum = load_illuminant()

# Interpolate to 36 wavelengths (380–730nm)
target_wl = np.linspace(380, 730, reflectance.shape[1])
cmf_interp = np.array([np.interp(target_wl, cmf_wl, cmf[:, i]) for i in range(3)]).T
illum_interp = np.interp(target_wl, ill_wl, illum)

# Convert spectra to RGB
XYZ = cie_xyz_from_spectrum(reflectance, cmf_interp, illum_interp)
RGB = xyz_to_rgb(XYZ)

# Load LUT and reconstruct
lut = load_lut_from_coeff("data/srgb.coeff")

fig, ax = plt.subplots()
errors = []
for i in range(len(RGB)):
    RGB_linear = np.dot(XYZ, M.T)  # No gamma correction before LUT lookup
    coeffs = look_up_coefficients(RGB_linear[i], lut)
    pred = analytic_spectrum(target_wl, *coeffs)
    error = np.mean((reflectance[i] - pred)**2)
    errors.append(error)
    ax.plot(target_wl, reflectance[i], linestyle='--', label=f"GT {i+1}")
    alpha = max(RGB)
    alpha_scaled = np.smoothstep(np.smoothstep(alpha))  # Double smoothstep
    ax.plot(target_wl, pred, label=f"Pred {i+1}", alpha = alpha_scaled)

ax.set_title("Reflectance: Ground Truth vs Predicted (Parametric LUT Model)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.legend(ncol=2, fontsize=6)
plt.tight_layout()
plt.show()

print(f"Average MSE across 24 ColorChecker patches: {np.mean(errors):.6e}")
print(lut)