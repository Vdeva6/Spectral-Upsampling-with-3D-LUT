import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

from src.lut_handler import load_lut

# === Wavelength Range ===
wavelengths = np.arange(380, 781, 10)  # 400–780nm in 10nm steps (41 values)

# === Spectral Model ===
def S(x):
    return 0.5 + (x / (2 * np.sqrt(1 + x**2)))

def spectrum(gamma, c0, c1, c2):
    x = (c0 * gamma**2) + (c1 * gamma) + c2
    return S(x)

def load_lut_from_file(path='data/lut.npy'):
    return np.load(path)


# === Trilinear Interpolation for Coefficient Lookup ===
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

    # Retrieve LUT corners
    V000 = lut[i0, j0, k0]
    V100 = lut[i1, j0, k0]
    V010 = lut[i0, j1, k0]
    V110 = lut[i1, j1, k0]
    V001 = lut[i0, j0, k1]
    V101 = lut[i1, j0, k1]
    V011 = lut[i0, j1, k1]
    V111 = lut[i1, j1, k1]

    # Interpolate
    R1 = lerp(V000, V100, fr)
    R2 = lerp(V010, V110, fr)
    R3 = lerp(V001, V101, fr)
    R4 = lerp(V011, V111, fr)
    G1 = lerp(R1, R2, fg)
    G2 = lerp(R3, R4, fg)
    return lerp(G1, G2, fb)

# === Load LUT and Render Spectrum ===
RGB = [0.2, 0.5, 0.7]
lut = load_lut('data/generated_lut.npy')
c0, c1, c2 = look_up_coefficients(RGB, lut)

# === Plot Result ===
fig, ax = plt.subplots()
ax.plot(wavelengths, spectrum(wavelengths, c0, c1, c2), color='green')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Spectral Curve from Dummy LUT')
fig.savefig("plots/basicplot.pdf")
plt.show()
