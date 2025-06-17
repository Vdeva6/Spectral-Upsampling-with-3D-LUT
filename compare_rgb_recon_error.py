import numpy as np
import matplotlib.pyplot as plt
from parse_colorchecker import parse_colorchecker
from parse_coeff_table import load_lut_from_coeff

# === Utilities ===

def load_cmf(path="data/cie-cmf.txt"):
    data, wavelengths = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = list(map(float, line.strip().split()))
            wavelengths.append(parts[0])
            data.append(parts[1:4])
    return np.array(data), np.array(wavelengths)

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

def lerp(a, b, f): return (1 - f) * a + f * b

def look_up_coefficients(rgb, lut):
    r, g, b = rgb
    scale = lut.shape[0] - 1
    r_idx, g_idx, b_idx = r * scale, g * scale, b * scale
    i0, j0, k0 = int(np.floor(r_idx)), int(np.floor(g_idx)), int(np.floor(b_idx))
    i1, j1, k1 = min(i0+1, scale), min(j0+1, scale), min(k0+1, scale)
    fr, fg, fb = r_idx - i0, g_idx - j0, b_idx - k0
    V000 = lut[i0, j0, k0]
    V100 = lut[i1, j0, k0]
    V010 = lut[i0, j1, k0]
    V110 = lut[i1, j1, k0]
    V001 = lut[i0, j0, k1]
    V101 = lut[i1, j0, k1]
    V011 = lut[i0, j1, k1]
    V111 = lut[i1, j1, k1]
    R1, R2 = lerp(V000, V100, fr), lerp(V010, V110, fr)
    R3, R4 = lerp(V001, V101, fr), lerp(V011, V111, fr)
    G1, G2 = lerp(R1, R2, fg), lerp(R3, R4, fg)
    return lerp(G1, G2, fb)

def S(x): return 0.5 + 0.5 * x * (1 / np.sqrt(1 + x**2))

def analytic_spectrum(wavelengths, c0, c1, c2):
    return S(c0 * wavelengths**2 + c1 * wavelengths + c2)

# === Main Evaluation ===

def main():
    reflectance, ref_wl = parse_colorchecker("data/ColorChecker_spectra.txt")
    cmf, cmf_wl = load_cmf()
    ill_wl, illum = load_illuminant()

    # Resample to 36 wavelengths
    target_wl = np.linspace(380, 730, reflectance.shape[1])
    cmf_interp = np.array([np.interp(target_wl, cmf_wl, cmf[:, i]) for i in range(3)]).T
    illum_interp = np.interp(target_wl, ill_wl, illum)

    # Convert GT reflectance → RGB
    XYZ_gt = cie_xyz_from_spectrum(reflectance, cmf_interp, illum_interp)
    RGB_gt = xyz_to_rgb(XYZ_gt)

    # Load LUT and reconstruct
    lut = load_lut_from_coeff("data/srgb.coeff")
    errors = []

    for i, rgb in enumerate(RGB_gt):
        coeffs = look_up_coefficients(rgb, lut)
        pred_refl = analytic_spectrum(target_wl, *coeffs)
        XYZ_pred = cie_xyz_from_spectrum(pred_refl, cmf_interp, illum_interp)
        RGB_pred = xyz_to_rgb(XYZ_pred)
        mse = np.mean((RGB_gt[i] - RGB_pred)**2)
        errors.append(mse)

    print(f"\n🎨 Average RGB-space MSE: {np.mean(errors):.6e}")
    for i, e in enumerate(errors, 1):
        print(f"Patch {i:2d} MSE: {e:.6f}")

if __name__ == "__main__":
    main()
