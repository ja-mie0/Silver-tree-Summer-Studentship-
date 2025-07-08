import os
import matplotlib.pyplot as plt

def read_spectral_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find the line where spectral data starts
    start_idx = next(i for i, line in enumerate(lines) if '>>>>>Begin Spectral Data<<<<<' in line) + 1

    wavelengths = []
    intensities = []

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                w = float(parts[0])
                i = float(parts[1])
                if w >= 0 and i >= 0:
                    wavelengths.append(w)
                    intensities.append(i)
            except ValueError:
                continue  # Skip lines that don't contain valid floats

    return wavelengths, intensities

def plot_spectral_files(directory, file_pattern="*.txt"):
    import glob
    from collections import defaultdict
    import numpy as np

    plt.figure(figsize=(12, 6))
    files = glob.glob(os.path.join(directory, file_pattern))
    files_dict = defaultdict(dict)

    # Group files by number and type (with/without)
    def extract_decimal_key(base):
        # Extract the first sequence of digits and insert decimal after first digit
        import re
        match = re.search(r'(\d+)', base)
        if match:
            digits = match.group(1)
            if len(digits) > 1:
                return digits[0] + '.' + digits[1:]
            else:
                return digits
        return base

    for file in files:
        base = os.path.basename(file)
        #print(base)
        base_clean = base.replace('without', 'without').replace(' .txt', '.txt').replace('.txt', '').replace('without', 'without').strip()
        if 'without' in base:
            key_raw = base.replace('without', '').replace('.txt', '').replace(' .txt', '').replace('without', '').strip()
            key = extract_decimal_key(key_raw)
            files_dict[key]['without'] = file
            #print(f"Added without file: {file}")
        else:
            key_raw = base.replace('.txt', '').replace(' .txt', '').strip()
            key = extract_decimal_key(key_raw)
            files_dict[key]['with'] = file
            #print(f"Added with file: {file}")

    # Assign a color to each key for consistent coloring
    import itertools
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    key_to_color = {}
    for file in files:
        base = os.path.basename(file)
        if 'without' in base:
            key_raw = base.replace('without', '').replace('.txt', '').replace(' .txt', '').replace('without', '').strip()
        else:
            key_raw = base.replace('.txt', '').replace(' .txt', '').strip()
        key = extract_decimal_key(key_raw)
        if key not in key_to_color:
            key_to_color[key] = next(color_cycle)

    # Plot all original spectra (filtered to 250-800 nm)
    for file in files:
        base = os.path.basename(file)
        if 'without' in base:
            key_raw = base.replace('without', '').replace('.txt', '').replace(' .txt', '').replace('without', '').strip()
        else:
            key_raw = base.replace('.txt', '').replace(' .txt', '').strip()
        key = extract_decimal_key(key_raw)
        label = f"{key} {'without' if 'without' in base else 'with'}"
        color = key_to_color[key]
        wavelengths, intensities = read_spectral_data(file)
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)
        mask = (wavelengths >= 250) & (wavelengths <= 800)
        wavelengths = wavelengths[mask]
        intensities = intensities[mask]
        plt.plot(wavelengths, intensities, label=label, color=color)
    plt.title("Spectral Data")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity compared to a white standard (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot difference curves (with - without), filtered to 250-800 nm
    plt.figure(figsize=(12, 6))
    for key, pair in files_dict.items():
        if 'with' in pair and 'without' in pair:
            w_with, i_with = read_spectral_data(pair['with'])
            w_without, i_without = read_spectral_data(pair['without'])
            w_with = np.array(w_with)
            i_with = np.array(i_with)
            w_without = np.array(w_without)
            i_without = np.array(i_without)
            # Interpolate 'without' to 'with' wavelengths if needed
            if not np.array_equal(w_with, w_without):
                i_without_interp = np.interp(w_with, w_without, i_without)
            else:
                i_without_interp = i_without
            diff = i_without_interp-i_with
            mask = (w_with >= 250) & (w_with <= 800)
            color = key_to_color.get(key, None)
            plt.plot(w_with[mask], diff[mask], label=f"{key} (without - with)", color=color)
    plt.title("Difference Spectra (without index matching fluid minus with)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot ratio curves (without / with), filtered to 250-800 nm
    plt.figure(figsize=(12, 6))
    for key, pair in files_dict.items():
        if 'with' in pair and 'without' in pair:
            w_with, i_with = read_spectral_data(pair['with'])
            w_without, i_without = read_spectral_data(pair['without'])
            w_with = np.array(w_with)
            i_with = np.array(i_with)
            w_without = np.array(w_without)
            i_without = np.array(i_without)
            if not np.array_equal(w_with, w_without):
                i_without_interp = np.interp(w_with, w_without, i_without)
            else:
                i_without_interp = i_without
            i_with_arr = np.array(i_with)
            # Avoid division by zero
            with_nonzero = np.where(i_with_arr == 0, np.nan, i_with_arr)
            ratio = np.array(i_without_interp) / with_nonzero
            mask = (w_with >= 250) & (w_with <= 800)
            color = key_to_color.get(key, None)
            plt.plot(w_with[mask], ratio[mask], label=f"{key} (without / with)", color=color)
    plt.title("Ratio Spectra (without / with)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
plot_spectral_files("optical indexing")  # Plots all .txt files in the directory
