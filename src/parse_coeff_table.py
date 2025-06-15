import numpy as np

def load_lut_from_coeff(path="data/srgb.coeff"):
    raw = np.fromfile(path, dtype=np.float32)
    for r in raw:
        print (r)
    # Total values expected in a 33³ LUT with 3 coefficients each
    expected_count = 33 * 33 * 33 * 3

    # Scan for a valid starting index by checking if slice can reshape
    for start in range(len(raw) - expected_count):
        try:
            sub = raw[start:start + expected_count]
            reshaped = sub.reshape((33, 33, 33, 3))
            return reshaped
        except ValueError:
            continue

    raise ValueError("Could not find valid LUT data block in srgb.coeff.")
