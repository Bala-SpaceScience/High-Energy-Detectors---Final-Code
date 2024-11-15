import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_spectrum(channels, counts, roi, initial_params, bounds):
    in_roi = (channels >= roi[0]) & (channels <= roi[1])
    channels_roi = channels[in_roi]
    counts_roi = counts[in_roi]
    popt, pcov = curve_fit(gaussian, channels_roi, counts_roi, p0=initial_params, bounds=bounds)
    return popt, pcov, channels_roi, counts_roi

def load_spectrum(file_path, scale=1.0):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    counts = []
    for line in lines:
        try:
            count = int(line.strip())
            counts.append(count)
        except ValueError:        
            continue

    counts = np.array(counts) * scale
    return counts

# Defining the line eqn for the given parameters
def linear_calibration(x, a, b):
    return a * x + b

# Function to convert FWHM from channels to keV
def channel_to_energy(channel, a, b):
    return a * channel + b

# Defining the model R^2 = a*E^-2 + b*E^-1 + c
def resolution_model(E, a, b, c):
    return a * E**-2 + b * E**-1 + c

# Defining polynomial model for intrinsic efficiency
def efficiency_model(logE, a, b, c):
    return a + b * logE + c * logE**2

base_path = os.path.join(os.path.dirname(__file__), 'labdata', 'BGO Detector')

file_paths = {
    "Am241": os.path.join(base_path, 'BGO AM241 0deg 300sec.Spe'),
    "Ba133": os.path.join(base_path, 'BGO Ba133 0deg 8min.Spe'),
    "Co60": os.path.join(base_path, 'BGO Co60 0deg 600sec.Spe'), 
    "Cs137": os.path.join(base_path, 'BGO Cs137 0deg 300sec.Spe')
}

scaling = {
    "Am241": 1.0,
    "Ba133": 1.0,
    "Co60": 0.5,
    "Cs137": 1.0
}

spectra = {source: load_spectrum(path, scaling[source]) for source, path in file_paths.items()}

base_path_bg = os.path.join(os.path.dirname(__file__), 'labdata')
bg_noise_path = os.path.join(base_path_bg, 'BGO Background.Spe')


bg_noise = load_spectrum(bg_noise_path)

def count_plot():
    plt.figure(figsize=(7, 10)) 
    for i, (source, counts) in enumerate(spectra.items(), 1):
    # removing background noise using the background readings taken
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        plt.subplot(4, 1, i) 
        plt.plot(counts, label=f'{source} Spectrum', marker='_', c='#126199')
        plt.xlabel('Channel Number')
        plt.ylabel('Counts')
        plt.title(f'BGO Detector Spectrum for {source}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout() 
    plt.show()


def curve_plot(plot=True):
    measurement_time = {
    "Am241": 300,
    "Ba133": 480,
    "Co60": 600,
    "Cs137": 300
}

    roi_values = {
    "Am241": (0, 50),
    "Ba133": (150, 250),
    "Co60": (550, 650),
    "Cs137": (220, 410)
}

# Initial Gaussian parameters for each source
    initial_params_dict = {
    "Am241": [20, 2, max(spectra["Am241"][roi_values["Am241"][0]:roi_values["Am241"][1]])],
    "Ba133": [197, 2, max(spectra["Ba133"][roi_values["Ba133"][0]:roi_values["Ba133"][1]])],
    "Co60": [600, 2, max(spectra["Co60"][roi_values["Co60"][0]:roi_values["Co60"][1]])],
    "Cs137": [380, 5, max(spectra["Cs137"][roi_values["Cs137"][0]:roi_values["Cs137"][1]])]
}

    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

# Additional ROI and initial params for Co-60's second peak
    roi_values_second = {
    "Co60": (650, 750)
}
    initial_params_second = {
    "Co60": [700, 2, max(spectra["Co60"][roi_values_second["Co60"][0]:roi_values_second["Co60"][1]])]
}

    res = pd.DataFrame(index=["Am241", "Ba133", "Cs137", "Co60", "Co60 (second peak)"], columns=["Mean(mu)", "Standard deviation", "Amplitude", "Total count", "Count rate"])

# Process each spectrum
    for i, (source, counts) in enumerate(spectra.items(), 1):
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        channels = np.arange(len(counts))
        roi = roi_values[source]
        initial_params = initial_params_dict[source]
        popt, pcov, channels_roi, counts_roi = fit_gaussian_spectrum(
        channels, counts, roi, initial_params, bounds
    )
        mu, sigma, amplitude = popt
    
    # Calculating total counts and count rate
        total_counts = amplitude * sigma * np.sqrt(2 * np.pi)
        count_rate = total_counts / measurement_time[source]

    # Plotting Gaussian fit and count rate
        if (plot):
            plt.figure(figsize=(6, 5))
            plt.plot(channels, counts, label="Raw Data", color="blue", marker='_')
            plt.plot(
            channels_roi,
            gaussian(channels_roi, *popt),
            label="First Gaussian Fit",
            color="red",
            linestyle="--"
            )

    # Second peak for Co-60
        if source == "Co60":
            roi_second = roi_values_second[source]
            initial_params2 = initial_params_second[source]
            popt2, pcov2, channels_roi2, counts_roi2 = fit_gaussian_spectrum(
            channels, counts, roi_second, initial_params2, bounds
        )
            mu2, sigma2, amplitude2 = popt2
            total_counts2 = amplitude2 * sigma2 * np.sqrt(2 * np.pi)
            count_rate2 = total_counts2 / measurement_time[source]
        
            if (plot):
                plt.plot(
                channels_roi2,
                gaussian(channels_roi2, *popt2),
                label="Second Gaussian Fit",
                color="red",
                linestyle="--"
                )

            res.loc["Co60 (second peak)"] = [round(mu2, 3), round(sigma2, 3), round(amplitude2, 3), round(total_counts2, 3), round(count_rate2, 3)]

        if (plot):
            plt.xlabel("Channel Number")
            plt.ylabel("Counts")
            plt.title(f"Gaussian Fit for {source} - BGO Detector")
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
    "Co60": 1173.228,
    "Co60_second": 1332.429,
    "Cs137": 661.657
    }
    mu_values = {
    "Am241": res["Mean(mu)"]["Am241"],
    "Ba133": res["Mean(mu)"]["Ba133"],
    "Co60": res["Mean(mu)"]["Co60"],
    "Co60_second": res["Mean(mu)"]["Co60 (second peak)"],
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
    plt.title("Energy Calibration Curve for BGO Detector")
    plt.legend()
    plt.grid(True)
    plt.show()

    example_channel = 500
    print(f"Energy at channel {example_channel}: {channel_to_energy(example_channel, a, b):.2f} keV")

def resolution_plot(res):
    # Plotting peaks from Am, Ba and Co
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Ba133": 356.0129,
        "Co60": 1173.228,
    }

    # Standard deviations obtained from Gaussian fitting
    sigma_values = {
        "Am241": res["Standard deviation"]["Am241"],
        "Ba133": res["Standard deviation"]["Ba133"],
        "Co60": res["Standard deviation"]["Co60"],
        "Co60_second": res["Standard deviation"]["Co60 (second peak)"],
        "Cs137": res["Standard deviation"]["Cs137"]
    }

    # Calibration constants from linear fit
    a, b = 1.947, 18.200  

    fwhm_keV = {}
    energy_resolution = {}

    for source, energy in known_energies.items():
        sigma = sigma_values[source]
        fwhm = 2.355 * sigma
        fwhm_keV_value = channel_to_energy(fwhm, a, b) - channel_to_energy(0, a, b) 
        fwhm_keV[source] = fwhm_keV_value
        energy_resolution[source] = fwhm_keV_value / energy 

    print("FWHM (keV) and Energy Resolution (R) for each source:")
    for source in known_energies:
        print(f"{source}: FWHM = {fwhm_keV[source]:.2f} keV, R = {energy_resolution[source]:.4f}")

    # Preparing data for fitting the model
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
    plt.title("Energy Resolution vs. Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display fitted model parameters
    print(f"Fitted Resolution Model: R^2 = {a_fit:.4f} * E^-2 + {b_fit:.4f} * E^-1 + {c_fit:.4f}")


def efficiency_plot(res):
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Ba133": 356.0129,
        "Co60": 1173.228,
        "Cs137": 661.657
    }

    # Source activities for each source (in Bq, or disintegrations per second)
    source_activities = {
        "Am241": 474340,
        "Ba133": 19938.19,
        "Co60": 1039.7,
        "Cs137": 160580
    }

    # Count rates (counts per second) for each source from gaussian fitting
    count_rates = {
        "Am241": res["Count rate"]["Am241"],
        "Ba133": res["Count rate"]["Ba133"],
        "Co60": res["Count rate"]["Co60"],
        "Co60_second": res["Count rate"]["Co60 (second peak)"],
        "Cs137": res["Count rate"]["Cs137"]
    }

    # Geometry factor for intrinsic efficiency calculation
    geometry_factor = 0.037

    # Calculate absolute and intrinsic efficiencies using count rates directly
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

    # Plotting the intrinsic efficiency vs. energy on a linear-linear scale
    plt.figure(figsize=(8, 6))
    plt.scatter(energies, intrinsic_eff_values, color='blue', label="Intrinsic Efficiency")
    plt.plot(energy_range, smooth_efficiency, 'r--', label="Fitted Efficiency")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intrinsic Efficiency (ε)")
    plt.title("Intrinsic Efficiency vs. Energy for BGO Detector" )
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nFitted Efficiency Model: ln(ε) = {a_fit:.4f} + {b_fit:.4f} * ln(E) + {c_fit:.4f} * (ln(E))^2")

def off_axis_plot():
    # Define the file paths for the angle measurements
    base_path_1 = os.path.join(os.path.dirname(__file__), 'labdata')

# Define file paths using the base path
    file_paths = {
        "0deg":  os.path.join(base_path_1, 'BGO AM241 0deg 300sec.Spe'),
        "30deg": os.path.join(base_path_1, 'BGO AM241 30deg 300sec.Spe'),
        "60deg": os.path.join(base_path_1, 'BGO AM241 60deg 300sec.Spe'),
        "90deg": os.path.join(base_path_1, 'BGO AM241 90deg 300sec.Spe')
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
    plt.title("Counts vs. Channel for Different Angles in BGO Detector")
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

    total_emitted_counts = source_activity * measurement_time

    # Calculating efficiency metrics for each angle
    efficiency_metrics = {}
    for angle, count in total_counts.items():
        angle_deg = int(angle.replace("deg", ""))
        # Geo factor calculated manually
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
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for BGO Detector")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting intrinsic efficiency as a function of angle
    plt.figure(figsize=(7, 6))
    plt.plot(angles, intrinsic_efficiencies, marker='o', linestyle='--', color='green', label="Intrinsic Efficiency")
    plt.xlabel("Off-Axis Angle (degrees)")
    plt.ylabel("Efficiency / Geometric Factor")
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for BGO Detector")
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