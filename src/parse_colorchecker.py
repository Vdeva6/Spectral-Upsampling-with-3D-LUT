import numpy as np

def parse_colorchecker(path="data/ColorChecker_spectra.txt"):
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
            continue  # Skip invalid lines
        try:
            spectrum = [float(x) for x in parts[2:]]  # Skip index and label
            spectra.append(spectrum)
        except ValueError:
            continue  # Skip lines with invalid data

    data = np.array(spectra)
    wavelengths = np.linspace(380, 730, data.shape[1])
    return data, wavelengths
