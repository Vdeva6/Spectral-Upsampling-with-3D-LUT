import numpy as np
from parse_colorchecker import parse_colorchecker
from compare_error import load_cmf, load_illuminant
import matplotlib.pyplot as plt

def cie_xyz_from_spectrum(reflectance, cmf, illuminant):
    k = 100 / np.sum(illuminant * cmf[:, 1])
    print("k:", k)
    print("Illuminant (first 5):", illuminant[:5])
    print("CMF Y (first 5):", cmf[:5, 1])
    print("Dot product shape:", np.dot(reflectance * illuminant[np.newaxis, :], cmf).shape)

    return k * np.dot(reflectance * illuminant[np.newaxis, :], cmf)

def xyz_to_srgb(XYZ):
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    RGB = np.dot(XYZ, M.T)

    # Clamp negative values to 0 before gamma correction
    RGB = np.clip(RGB, 0, None)

    def gamma_correct(c):
        return np.where(c <= 0.0031308,
                        12.92 * c,
                        1.055 * np.power(c, 1 / 2.4) - 0.055)
    
    RGB = gamma_correct(RGB)

    # Final clamp to [0, 1]
    return np.clip(RGB, 0, 1)


def main():
    # Load spectral data
    reflectances, wavelengths = parse_colorchecker("data/ColorChecker_spectra.txt")  # (24, 36)

    cmf, cmf_wl = load_cmf("data/cie-cmf.txt")               # (81, 3), (81,)
    illuminant, ill_wl = load_illuminant("data/D65.5nm")     # (81,), (81,)

    # Interpolate both to match ColorChecker wavelengths
    target_wl = wavelengths  # [36]
    cmf_interp = np.array([np.interp(target_wl, cmf_wl, cmf[:, i]) for i in range(3)]).T
    illum_interp = np.interp(target_wl, ill_wl, illuminant)

    # Convert to XYZ
    XYZs = cie_xyz_from_spectrum(reflectances, cmf_interp, illum_interp)

    # Convert to sRGB
    print("🧮 Raw XYZs (first 5 patches):")
    print(XYZs[:5])
    sRGBs = xyz_to_srgb(XYZs)
    sRGBs = sRGBs[:24]

    # Plot as a color swatch
    fig, ax = plt.subplots(figsize=(12, 2))
    for i, rgb in enumerate(sRGBs):
        color = np.clip(rgb, 0, 1)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(sRGBs))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("ColorChecker RGB (from Reflectance Spectra)")
    plt.show()


    # BabelColor Classic 24 reference sRGB values under D65, normalized to [0, 1]
    reference_sRGB = np.array([
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67],
    [133, 128, 177], [103, 189, 170], [214, 126,  44], [ 80,  91, 166],
    [193,  90,  99], [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31],
    [187,  86, 149], [  8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52]
    ]) / 255.0

    # Compute per-patch MSE
    mse_per_patch = np.mean((sRGBs - reference_sRGB) ** 2, axis=1)
    avg_mse = np.mean(mse_per_patch)

    # Print MSEs
    for i, mse in enumerate(mse_per_patch):
        print(f"Patch {i+1:2d} MSE: {mse:.6f}")
        print(f"\n📊 Average sRGB MSE vs. Reference: {avg_mse:.6f}")

    # --- Visual Comparison Plot ---
    fig, ax = plt.subplots(2, 1, figsize=(12, 2))
    for i in range(24):
        ax[0].add_patch(plt.Rectangle((i, 0), 1, 1, color=reference_sRGB[i]))
        ax[1].add_patch(plt.Rectangle((i, 0), 1, 1, color=sRGBs[i]))

        ax[0].set_xlim(0, 24)
        ax[1].set_xlim(0, 24)
        ax[0].set_ylim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[0].set_title("🎨 Reference sRGB (ColorChecker)")
        ax[1].set_title("🧪 Reconstructed sRGB from Spectral")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
