import numpy as np
import matplotlib.pyplot as plt

# -------------------- Load Functions --------------------

def load_colorchecker_spectra(path="data/ColorChecker_spectra.txt"):
    with open(path, "r") as f:
        lines = f.readlines()

    # Locate BEGIN_DATA
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("BEGIN_DATA"):
            start = i + 2  # Skip header lines
            break

    spectra = []
    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            spectrum = [float(x) for x in parts[2:]]  # Skip index and label
            spectra.append(spectrum)
        except ValueError:
            continue

    data = np.array(spectra)
    wavelengths = np.linspace(380, 730, data.shape[1])
    return data, wavelengths

def load_cmf(path="data/cie-cmf.txt"):
    data = np.loadtxt(path)
    wl = data[:, 0]
    cmf = data[:, 1:4]  # columns for x̄, ȳ, z̄
    return wl, cmf

def load_illuminant(path="data/D65.5nm"):
    data = np.loadtxt(path)
    wl = data[:, 0]
    illum = data[:, 1]
    return wl, illum

# -------------------- Color Conversion --------------------

def cie_xyz_from_spectrum(reflectance, cmf, illuminant):
    k = 100 / np.sum(cmf[:, 1] * illuminant)
    return k * np.dot(reflectance * illuminant[np.newaxis, :], cmf)

def xyz_to_rgb(XYZ):
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    rgb = np.dot(XYZ, M.T)
    return np.clip(rgb / np.max(rgb), 0, 1)

# -------------------- Main Pipeline --------------------

# Load reflectance data
reflectance, target_wl = load_colorchecker_spectra()

# Load and interpolate CMFs and D65 to match reflectance wavelengths
cmf_wl, cmf = load_cmf()
illum_wl, illum = load_illuminant()

cmf_interp = np.stack([
    np.interp(target_wl, cmf_wl, cmf[:, 0]),
    np.interp(target_wl, cmf_wl, cmf[:, 1]),
    np.interp(target_wl, cmf_wl, cmf[:, 2])
], axis=1)  # shape: [N wavelengths x 3 channels]

illum_interp = np.interp(target_wl, illum_wl, illum)

# Convert each patch
XYZ = cie_xyz_from_spectrum(reflectance, cmf_interp, illum_interp)
RGB = xyz_to_rgb(XYZ)

# -------------------- Render Color Patches --------------------

fig, ax = plt.subplots(figsize=(12, 2))
for i, rgb in enumerate(RGB):
    ax.bar(i, 1, color=rgb, width=0.9)

ax.set_xlim(-1, len(RGB))
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("RGB Colors Converted from ColorChecker Spectra")
plt.tight_layout()
plt.show()
