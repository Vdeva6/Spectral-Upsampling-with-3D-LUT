import numpy as np
from parse_colorchecker import parse_colorchecker
from og_compare_error import load_cmf, load_illuminant, cie_xyz_from_spectrum, xyz_to_rgb, srgb_gamma_encode

# 1. Load reflectance data and wavelengths
reflectance, wavelengths = parse_colorchecker()

# 2. Load CMF and Illuminant
cmf_wl, cmf = load_cmf()
ill_wl, illum = load_illuminant()

# 3. Interpolate to target wavelengths (ColorChecker range)
cmf_interp = np.array([np.interp(wavelengths, cmf_wl, cmf[:,i]) for i in range(3)]).T
illum_interp = np.interp(wavelengths, ill_wl, illum)

# 4. Convert to XYZ and then to linear RGB
XYZ = cie_xyz_from_spectrum(reflectance, cmf_interp, illum_interp)
RGB_linear = xyz_to_rgb(XYZ)

# 5. Encode to sRGB
RGB_encoded = srgb_gamma_encode(RGB_linear)

# 6. Print first few RGB values (linear + encoded)
for i in range(5):
    print(f"Patch {i+1}:")
    print("  Linear RGB :", RGB_linear[i])
    print("  Encoded RGB:", RGB_encoded[i])
