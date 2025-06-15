import numpy as np
import os

# Path to save/load the LUT
LUT_PATH = os.path.join("data", "generated_lut.npy")


def generate_dummy_lut(resolution=33):
    """
    Generates a dummy LUT (lookup table) of the specified resolution.
    Each entry is a 3-tuple (c0, c1, c2) used to compute reflectance spectrum.

    Parameters:
        resolution (int): Size of each LUT axis (default is 33 for 33x33x33)

    Returns:
        np.ndarray: LUT of shape (resolution, resolution, resolution, 3)
    """
    lut = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                r = i / (resolution - 1)
                g = j / (resolution - 1)
                b = k / (resolution - 1)
                # Example: coefficients linearly scaled from RGB
                c0 = -1e-5 * r
                c1 = 1e-2 * g
                c2 = -6 + 1.5 * b
                lut[i, j, k] = [c0, c1, c2]

    return lut


def save_lut(lut, path=LUT_PATH):
    """Saves the LUT to a .npy file."""
    np.save(path, lut)
    print(f"LUT saved to {path}")


def load_lut(path=LUT_PATH):
    """Loads the LUT from a .npy file."""
    return np.load(path)


if __name__ == "__main__":
    # Run this file directly to regenerate and save the LUT
    lut = generate_dummy_lut()
    save_lut(lut)
