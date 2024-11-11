import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os


# Function to load data from a .Spe file, skipping non-numeric lines
def load_spectrum(file_path, scale=1.0):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Extract only the numeric data, ignoring headers and other text
    counts = []
    for line in lines:
        try:
            count = int(line.strip())
            counts.append(count)
        except ValueError:
            continue
    counts = np.array(counts) * scale
    return counts

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Function to fit Gaussian and calculate count rate
def fit_gaussian_spectrum(channels, counts, roi, initial_params, bounds, acquisition_time):
    in_roi = (channels >= roi[0]) & (channels <= roi[1])
    channels_roi = channels[in_roi]
    total_counts = counts[in_roi]

    popt, pcov = curve_fit(gaussian, channels_roi, total_counts, p0=initial_params, bounds=bounds)
    mu, sigma, amplitude = popt
    
    # Calculating FWHM and count rate
    fwhm = 2.355 * sigma
    total_counts = amplitude * fwhm
    count_rate = total_counts / acquisition_time
    
    return popt, pcov, channels_roi, total_counts, count_rate

# Line eqution for the given parameters
def linear_calibration(x, a, b):
    return a * x + b

# Function to convert FWHM from channels to keV
def channel_to_energy(channel, a, b):
    return a * channel + b

# Define the model R^2 = a*E^-2 + b*E^-1 + c
def resolution_model(E, a, b, c):
    return a * E**-2 + b * E**-1 + c

# Polynomial model for intrinsic efficiency
def efficiency_model(logE, a, b, c):
    return a + b * logE + c * logE**2


# File paths for each source's .Spe file
base_path = os.path.join(os.path.dirname(__file__), 'labdata', 'NaI detector')

file_paths = {
    "Am241": os.path.join(base_path, 'NaI AM241 0deg 300sec.Spe'),
    "Ba133": os.path.join(base_path, 'NaI Ba133 0deg 300sec.Spe'),
    "Co60": os.path.join(base_path, 'NaI Co60 0deg 600sec.Spe'),
    "Cs137": os.path.join(base_path, 'NaI Cs137 0deg 300sec.Spe')
}

# Scaling factor for Co-60 to match the 300s duration of the other files
scaling = {
    "Am241": 1.0,
    "Ba133": 1.0,
    "Co60": 0.5,
    "Cs137": 1.0
}


bg_noise_path = os.path.join(base_path, 'NaI Background.Spe')
bg_noise = load_spectrum(bg_noise_path)

spectra = {source: load_spectrum(path, scaling[source]) for source, path in file_paths.items()}

# Loading data for each source with scaling applied and noise removed
def count_plot():
    plt.figure(figsize=(7, 10))
    for i, (source, counts) in enumerate(spectra.items(), 1):
        plt.subplot(4, 1, i)  
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        plt.plot(counts, label=f'{source} Spectrum')
        plt.xlabel('Channel Number')
        plt.ylabel('Counts')
        plt.title(f'NAI Detector Spectrum for {source}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def curve_plot(plot=True):
    # ROIs and initial parameters for fitting
    roi_values = {
        "Am241": (20, 50),
        "Ba133": (140, 200),
        "Co60": (23, 120),
        "Cs137": (220, 410)
    }

    initial_params_dict = {
        "Am241": [20, 2, max(spectra["Am241"][roi_values["Am241"][0]:roi_values["Am241"][1]])],
        "Ba133": [197, 200, max(spectra["Ba133"][roi_values["Ba133"][0]:roi_values["Ba133"][1]])],
        "Co60": [85, 2, max(spectra["Co60"][roi_values["Co60"][0]:roi_values["Co60"][1]])],
        "Cs137": [380, 5, max(spectra["Cs137"][roi_values["Cs137"][0]:roi_values["Cs137"][1]])]
    }

    measurement_times = {
        "Am241": 300,
        "Ba133": 300,
        "Co60": 600,
        "Cs137": 300
    }

    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    res = pd.DataFrame(index=["Am241", "Ba133", "Cs137"], columns=["Mean(mu)", "Standard deviation", "Amplitude", "Total count", "Count rate"])

    # Loop to fit Gaussian, plot results, and calculate count rates
    for source, counts in spectra.items():
        # Skipping Co60 since coudln't fit the proper photo peak
        if source == "Co60":
            continue 
        
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        
        channels = np.arange(len(counts))
        roi = roi_values[source]
        initial_params = initial_params_dict[source]
        acquisition_time = measurement_times[source]
        
        # Fitting Gaussian and calculate count rate
        popt, pcov, channels_roi, total_counts, count_rate = fit_gaussian_spectrum(
            channels, counts, roi, initial_params, bounds, acquisition_time
        )
        mu, sigma, amplitude = popt
        
        if (plot):
            # Plotting the raw data and fitted Gaussian
            plt.figure(figsize=(8, 6))
            plt.plot(channels, counts, label="Raw Data", color="blue", marker='_')
            plt.plot(
                channels_roi,
                gaussian(channels_roi, *popt),
                label="Gaussian Fit",
                color="red",
                linestyle="--"
            )
            plt.xlabel("Channel Number")
            plt.ylabel("Counts")
            plt.title(f"Gaussian Fit of {source} for NaI")
            plt.legend()
            plt.grid(True)
            plt.show()

        res.loc[source] = [round(mu, 3), round(sigma, 3), round(amplitude, 3), round(total_counts, 3), round(count_rate, 3)]
    print(res)
    return res

def calibration_plot(res):
    known_energies = {
        "Am241": 59.5409,
        "Ba133": 356.0129,
        "Cs137": 661.657
    }

    # Mean values obtained from Gaussian fitting
    mu_values = {
        "Am241": res["Mean(mu)"]["Am241"],
        "Ba133": res["Mean(mu)"]["Ba133"],
        "Cs137": res["Mean(mu)"]["Cs137"]
    }

    energies = list(known_energies.values())
    channels = list(mu_values.values())


    popt, pcov = curve_fit(linear_calibration, channels, energies)
    a, b = popt

    print(f"Calibration Line: Energy (keV) = {a:.3f} * Channel + {b:.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(channels, energies, color='blue', label="Known Peaks")
    plt.plot(np.arange(0, max(channels) * 1.1), 
            linear_calibration(np.arange(0, max(channels) * 1.1), *popt),
            color='red', label=f"Fitted Line: E = {a:.3f} * Channel + {b:.3f}")
    plt.xlabel("Channel Number")
    plt.ylabel("Energy (keV)")
    plt.title("Energy Calibration Curve for NaI Detector")
    plt.legend()
    plt.grid(True)
    plt.show()


    example_channel = 500
    print(f"Energy at channel {example_channel}: {channel_to_energy(example_channel, a, b):.2f} keV")

def resolution_plot(res):
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Ba133": 356.0129,
        "Cs137": 1173.228
    }

    # Standard deviations (sigma) obtained from Gaussian fitting
    sigma_values = {
        "Am241": res["Standard deviation"]["Am241"],
        "Ba133": res["Standard deviation"]["Ba133"],
        "Cs137": res["Standard deviation"]["Cs137"]
    } 

    # Calibration constants from linear fit
    a, b = 2.342, -7.314  

    # Calculate FWHM and energy resolution
    fwhm_keV = {}
    energy_resolution = {}

    for source, energy in known_energies.items():
        sigma = sigma_values[source]
        fwhm = 2.355 * sigma
        fwhm_keV_value = channel_to_energy(fwhm, a, b) - channel_to_energy(0, a, b)
        fwhm_keV[source] = fwhm_keV_value
        energy_resolution[source] = fwhm_keV_value / energy

    # Displaying FWHM and energy resolution results
    print("FWHM (keV) and Energy Resolution (R) for each source:")
    for source in known_energies:
        print(f"{source}: FWHM = {fwhm_keV[source]:.2f} keV, R = {energy_resolution[source]:.4f}")

    energies = np.array(list(known_energies.values()))
    resolutions_squared = np.array([energy_resolution[source]**2 for source in known_energies])


    # Fitting the model to the data
    popt, pcov = curve_fit(resolution_model, energies, resolutions_squared)
    a_fit, b_fit, c_fit = popt

    # Generating a smooth range of energy values for plotting
    energy_range = np.linspace(min(energies), max(energies), 500)  
    smooth_resolution = np.sqrt(resolution_model(energy_range, *popt))

    # Plotting the energy resolution vs. energy
    plt.figure(figsize=(8, 6))
    plt.scatter(energies, np.sqrt(resolutions_squared), color='blue', label="Resolution Data")
    plt.plot(energy_range, smooth_resolution, 'r--', label="Fitted Resolution Model")  
    plt.xlabel("Energy (keV)")
    plt.ylabel("Energy Resolution (R)")
    plt.title("Energy Resolution vs. Energy for NaI")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Fitted Resolution Model: R^2 = {a_fit:.4f} * E^-2 + {b_fit:.4f} * E^-1 + {c_fit:.4f}")

def efficiency_plot(res):
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Ba133": 356.0129,
        "Cs137": 661.657
    }

    # Source activities for each source (in Bq, or disintegrations per second)
    source_activities = {
        "Am241": 474340,
        "Ba133": 19938.19,
        "Cs137": 160580
    }

    # Count rates (counts per second) for each source from gaussian fitting
    count_rates = {
        "Am241": res["Count rate"]["Am241"],
        "Ba133": res["Count rate"]["Ba133"],
        "Cs137": res["Count rate"]["Cs137"],
    }

    # Geometry factor for intrinsic efficiency calculation
    geometry_factor = 0.037

    # Calculating absolute and intrinsic efficiencies using count rates directly
    absolute_efficiency = {}
    intrinsic_efficiency = {}

    for source in known_energies:
        if source in count_rates and source in source_activities:
            absolute_efficiency[source] = count_rates[source] / source_activities[source]
            intrinsic_efficiency[source] = absolute_efficiency[source] / geometry_factor
        else:
            print(f"Data missing for {source}, skipping efficiency calculation.")

    print("\nCalculated Absolute and Intrinsic Efficiencies:")
    for source in known_energies:
        if source in absolute_efficiency:
            print(f"{source}: Absolute Efficiency = {absolute_efficiency[source]:.6f}, Intrinsic Efficiency = {intrinsic_efficiency[source]:.6f}")

    # Polynomial fitting for ln(efficiency) vs ln(energy) for intrinsic efficiency
    energies = np.array([energy for source, energy in known_energies.items() if source in intrinsic_efficiency])
    intrinsic_eff_values = np.array([intrinsic_efficiency[source] for source in intrinsic_efficiency])

    log_energies = np.log(energies)
    log_intrinsic_eff = np.log(intrinsic_eff_values)


    # Fitting the model to the data
    popt, pcov = curve_fit(efficiency_model, log_energies, log_intrinsic_eff)
    a_fit, b_fit, c_fit = popt

    # Generating smooth curve for plotting on a linear scale
    energy_range = np.linspace(energies.min(), energies.max(), 500)
    smooth_efficiency = np.exp(efficiency_model(np.log(energy_range), *popt))

    plt.figure(figsize=(8, 6))
    plt.scatter(energies, intrinsic_eff_values, color='blue', label="Intrinsic Efficiency")
    plt.plot(energy_range, smooth_efficiency, 'r--', label="Fitted Efficiency")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intrinsic Efficiency (ε)")
    plt.title("Intrinsic Efficiency vs. Energy for NaI")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display fitted model parameters
    print(f"\nFitted Efficiency Model: ln(ε) = {a_fit:.4f} + {b_fit:.4f} * ln(E) + {c_fit:.4f} * (ln(E))^2")

def off_axis_plot():
    # Define the file paths for the angle measurements
    base_path = os.path.join(os.path.dirname(__file__), 'labdata')

    file_paths = {
    "0deg": os.path.join(base_path, 'NaI AM241 0deg 300sec.Spe'),
    "30deg": os.path.join(base_path, 'NAl AM241 30deg 300sec.Spe'),
    "60deg": os.path.join(base_path, 'Nal AM241 60deg 300sec.Spe'),
    "90deg": os.path.join(base_path, 'NAl AM241 90deg 300sec.Spe')
}

    # Load counts for each angle
    counts_per_angle = {angle: load_spectrum(path) for angle, path in file_paths.items()}

    # Plot counts vs. channels for each angle
    plt.figure(figsize=(12, 8))
    for angle, counts in counts_per_angle.items():
        channels = np.arange(len(counts))
        plt.plot(channels, counts, label=f"{angle}")

    plt.xlabel("Channel Number")
    plt.ylabel("Counts")
    plt.xlim(0, 200)
    plt.title("Counts vs. Channel for Off-axis angles in NaI Detector")
    plt.legend(title="Angles")
    plt.grid(True)
    plt.show()


    # Calculate total counts for each angle by summing the counts in each channel
    total_counts = {angle: np.sum(counts) for angle, counts in counts_per_angle.items()}

    # Display the total counts for each angle
    print("Total counts for each angle:")
    for angle, count in total_counts.items():
        print(f"{angle}: {count}")

    # Parameters for efficiency calculations
    source_activity = 474340 
    measurement_time = 300 
    distance = 14.4 
    detector_radius = 2.54 

    total_counts = {angle: np.sum(counts) for angle, counts in counts_per_angle.items()}
    total_emitted_counts = source_activity * measurement_time

    # Calculating efficiency metrics for each angle
    efficiency_metrics = {}
    for angle, count in total_counts.items():
        angle_deg = int(angle.replace("deg", ""))
        geo_factor = 0.037

        abs_efficiency = count / total_emitted_counts
        intrinsic_efficiency = abs_efficiency / geo_factor
        
        efficiency_metrics[angle] = {
            "Total Counts": count,
            "Absolute Efficiency": abs_efficiency,
            "Intrinsic Efficiency": intrinsic_efficiency,
            "Geometric Factor": geo_factor
        }
    for angle, metrics in efficiency_metrics.items():
        print(f"{angle}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        print()

    angles = np.array([0, 30, 60, 90])
    absolute_efficiencies = [efficiency_metrics[f"{angle}deg"]["Absolute Efficiency"] for angle in angles]
    intrinsic_efficiencies = [efficiency_metrics[f"{angle}deg"]["Intrinsic Efficiency"] for angle in angles]
    geometric_factors = [efficiency_metrics[f"{angle}deg"]["Geometric Factor"] for angle in angles]

    # Plotting absolute efficiency as a function of angle
    plt.figure(figsize=(7, 6))
    plt.plot(angles, absolute_efficiencies, marker='o', linestyle='-', color='blue', label="Absolute Efficiency")
    plt.xlabel("Off-Axis Angle (degrees)")
    plt.ylabel("Efficiency / Geometric Factor")
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for NaI Detector")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting intrinsic efficiency as a function of angle
    plt.figure(figsize=(7, 6))
    plt.plot(angles, intrinsic_efficiencies, marker='o', linestyle='--', color='green', label="Intrinsic Efficiency")
    plt.xlabel("Off-Axis Angle (degrees)")
    plt.ylabel("Efficiency / Geometric Factor")
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for NaI Detector")
    plt.legend()
    plt.grid(True)
    plt.show()

def all_plots():
    count_plot()
    res = curve_plot()
    calibration_plot(res=res)
    resolution_plot(res=res)
    efficiency_plot(res=res)
    off_axis_plot()