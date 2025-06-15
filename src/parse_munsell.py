import numpy as np

def load_munsell_data(path='data/real.dat'):
    data = []
    with open(path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue  # skip empty or malformed lines
            try:
                # convert everything except the first column to float
                spectrum = [float(x) for x in tokens[1:]]
                data.append(spectrum)
            except ValueError:
                continue  # skip lines with non-numeric data
    return np.array(data)
