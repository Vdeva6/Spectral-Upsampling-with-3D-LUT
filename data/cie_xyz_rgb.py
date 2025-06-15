import numpy as np
import pandas as pd

# Load Color Matching Functions (CMF) and D65 from pre-saved NumPy arrays
# These should match the wavelengths from 380 to 780 nm at 5nm intervals (81 points)
cmf_xyz = np.load('data/cie-cmf.npy')  # shape: (81, 3)
d65 = np.load('data/D65.5nm.npy')      # shape: (81,)
wavelengths = np.arange(380, 781, 5)   # 380 to 780 inclusive

# Calculate normalization factor for D65
k = 100 / np.sum(d65 * cmf_xyz[:, 1])


def reflectance_to_xyz(reflectance):
    """
    Convert a spectral reflectance (shape: (81,)) to CIE XYZ using D65.
    """
    X = k * np.sum(reflectance * d65 * cmf_xyz[:, 0])
    Y = k * np.sum(reflectance * d65 * cmf_xyz[:, 1])
    Z = k * np.sum(reflectance * d65 * cmf_xyz[:, 2])
    return np.array([X, Y, Z])


def xyz_to_srgb(xyz):
    """
    Convert a CIE XYZ value (D65 white point) to sRGB.
    """
    # Matrix to convert XYZ to linear RGB (D65)
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb_linear = M @ xyz / 100.0

    # Gamma correction
    def gamma_correct(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055

    rgb = np.array([gamma_correct(c) for c in rgb_linear])
    rgb = np.clip(rgb, 0, 1)
    return rgb


def reflectance_to_rgb(reflectance):
    """
    Full pipeline from reflectance spectrum to RGB color.
    """
    xyz = reflectance_to_xyz(reflectance)
    return xyz_to_srgb(xyz)


# Example usage:
if __name__ == "__main__":
    sample_reflectance = np.random.rand(len(wavelengths))  # Dummy reflectance
    rgb = reflectance_to_rgb(sample_reflectance)
    print("RGB:", rgb)
