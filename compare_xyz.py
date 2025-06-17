import numpy as np
import matplotlib.pyplot as plt
from compare_error import (
    parse_colorchecker,
    cie_xyz_from_spectrum,
    load_cmf,
    load_illuminant
)


def compute_error(XYZ1, XYZ2):
    return np.mean((XYZ1 - XYZ2) ** 2)


def main():
    # Target wavelengths used in ColorChecker and LUT (36 samples from 380–730nm)
    target_wl = np.linspace(380, 730, 36)

    # Load reflectance data [24 x 36]
    reflectances, target_wl = parse_colorchecker("data/ColorChecker_spectra.txt")


    # Load CMF data [81 x 3] and original wavelengths [81]
    cmf, cmf_wl = load_cmf("data/cie-cmf.txt")

    # Interpolate CMFs to match target wavelengths → [36 x 3]
    cmf_interp = np.array([
        np.interp(target_wl, cmf_wl, cmf[:, i]) for i in range(3)
    ]).T

    # Load illuminant (D65) and wavelengths
    illum, illum_wl = load_illuminant("data/D65.5nm")

    # Interpolate illuminant to match target wavelengths → [36]
    illum_interp = np.interp(target_wl, illum_wl, illum)

    # Debug: check dimensions
    #print("CMF interp shape:", cmf_interp.shape)         # should be (36, 3)
    #print("Illuminant interp shape:", illum_interp.shape) # should be (36,)
    #print("Reflectance shape:", reflectances.shape)       # should be (24, 36)

    # Normalize constant for integration
    k = 100 / np.sum(illum_interp * cmf_interp[:, 1])

    #print("📏 Normalization constant k:")
    #print(k)
    #print("💡 Illuminant (first 10 values):")
    #print(illum_interp[:10])
    #print("📈 CMF sample (first 3 rows):")
    #print(cmf_interp[:3])
    #print("🧪 Reflectance sample (first row):")
    #print(reflectances[0])

    # Compute XYZ values for each patch
    XYZs = []
    for i, reflectance in enumerate(reflectances):
        XYZ = k * np.dot(reflectance * illum_interp[:, np.newaxis], cmf_interp)  # [1 x 36] ⋅ [36 x 3] → [1 x 3]
        XYZs.append(XYZ)
        print(f"Patch {i} XYZ: {XYZ}")

    XYZs = np.array(XYZs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(XYZs[:, 0], label="X")
    ax.plot(XYZs[:, 1], label="Y")
    ax.plot(XYZs[:, 2], label="Z")
    ax.set_title("ColorChecker XYZ Values")
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Tristimulus Value")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
