import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# Function to load data from a .mca file
def load_spectrum(file_path, scale_factor=1.0):
    with open(file_path, 'rb') as file:
        lines = file.readlines()
    counts = []
    for line in lines:
        try:
            count = int(line.strip())
            counts.append(count)
        except ValueError:
            continue
    counts = np.array(counts) * scale_factor
    return counts

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def double_gaussian(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
    return (amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) +
            amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2))

def fit_gaussian_spectrum(channels, counts, roi, initial_params, bounds, measurement_time, double_peak=False):
    in_roi = (channels >= roi[0]) & (channels <= roi[1])
    channels_roi = channels[in_roi]
    counts_roi = counts[in_roi]

    if double_peak:
        popt, pcov = curve_fit(double_gaussian, channels_roi, counts_roi, p0=initial_params, bounds=bounds)
    else:
        popt, pcov = curve_fit(gaussian, channels_roi, counts_roi, p0=initial_params, bounds=bounds)

    if double_peak:
        sigma1, amp1, sigma2, amp2 = popt[1], popt[2], popt[4], popt[5]
        fwhm1 = 2.355 * sigma1
        fwhm2 = 2.355 * sigma2
        total_counts1 = amp1 * fwhm1
        total_counts2 = amp2 * fwhm2
        count_rate1 = total_counts1 / measurement_time
        count_rate2 = total_counts2 / measurement_time
        return popt, pcov, channels_roi, counts_roi, (count_rate1, count_rate2)
    else:
        sigma, amplitude = popt[1], popt[2]
        fwhm = 2.355 * sigma
        total_counts = amplitude * fwhm
        count_rate = total_counts / measurement_time
        return popt, pcov, channels_roi, counts_roi, count_rate

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


# File paths for each source's .mca file
file_paths = {
    "Am241": 'labdata/CDTE/CDTE_Detector_Am241_0deg_600sec.mca',
    "Ba133": 'labdata/CDTE/CDTE_Detector_Ba133_0deg_600.mca',
    "Co60":  'labdata/CDTE/CDTE_Detector_Co60_0deg_1200.mca', 
    "Cs137": 'labdata/CDTE/CDTE_Detector_Cs137_0deg_600.mca'
}

scaling_factors = {
    "Am241": 1.0,
    "Ba133": 1.0,
    "Co60": 0.5,
    "Cs137": 1.0
}

spectra = {source: load_spectrum(path, scaling_factors[source]) for source, path in file_paths.items()}
bg_noise = load_spectrum('labdata\CDTE\CDTE_Background.mca')

def count_plot():
    # Plotting each spectrum in a separate subplot with peak detection
    plt.figure(figsize=(7, 10))

    for i, (source, counts) in enumerate(spectra.items(), 1):
        channels = np.arange(len(counts))
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        
        plt.subplot(4, 1, i)
        plt.plot(counts, label=f'{source} Spectrum', color="blue")       

        plt.xlabel('Channel Number')
        plt.ylabel('Counts')
        plt.title(f'CDTE Detector Spectrum for {source}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def curve_plot(plot=True):
    file_paths = {
        "Am241": 'labdata/CDTE/CDTE_Detector_Am241_0deg_600sec.mca',
        "Ba133": 'labdata/CDTE/CDTE_Detector_Ba133_0deg_600.mca',
    }

    scaling_factors = {
        "Am241": 1.0,
        "Ba133": 1.0
    }

    spectra = {source: load_spectrum(path, scaling_factors[source]) for source, path in file_paths.items()}

    roi_values = {
        "Am241_main": (1170, 1200),
        "Am241_secondary": (200, 500),
        "Ba133": (600, 630)
    }

    initial_params_dict = {
        "Am241_main": [1185, 5, max(spectra["Am241"][roi_values["Am241_main"][0]:roi_values["Am241_main"][1]])],
        "Am241_secondary": [280, 5, max(spectra["Am241"][roi_values["Am241_secondary"][0]:roi_values["Am241_secondary"][1]]),
                            380, 5, max(spectra["Am241"][roi_values["Am241_secondary"][0]:roi_values["Am241_secondary"][1]])],
        "Ba133": [615, 5, max(spectra["Ba133"][roi_values["Ba133"][0]:roi_values["Ba133"][1]])]
    }

    bounds_single = ([0, 0, 0], [np.inf, np.inf, np.inf])
    bounds_double = ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    # measurement times for each source
    measurement_times = {
        "Am241": 600,
        "Ba133": 600
    }

    res = pd.DataFrame(index=["Am241", "Am241_secondary", "Am241_third", "Ba133"], columns=["Mean(mu)", "Standard deviation", "Amplitude", "Count rate"])
    for source, counts in spectra.items():
        
        adjusted_counts = counts - bg_noise
        counts = np.maximum(adjusted_counts, bg_noise)
        
        channels = np.arange(len(counts))
        measurement_time = measurement_times[source]
        
        if source == "Am241":
            # Main peak Gaussian fit
            roi_main = roi_values["Am241_main"]
            initial_params_main = initial_params_dict["Am241_main"]
            popt_main, pcov_main, channels_main, counts_main, count_rate_main = fit_gaussian_spectrum(
                channels, counts, roi_main, initial_params_main, bounds_single, measurement_time
            )
            mu_main, sigma_main, amplitude_main = popt_main

            # Double peak Gaussian fit for secondary peaks
            roi_secondary = roi_values["Am241_secondary"]
            initial_params_secondary = initial_params_dict["Am241_secondary"]
            popt_secondary, pcov_secondary, channels_secondary, counts_secondary, (count_rate1, count_rate2) = fit_gaussian_spectrum(
                channels, counts, roi_secondary, initial_params_secondary, bounds_double, measurement_time, double_peak=True
            )
            mu1, sigma1, amp1, mu2, sigma2, amp2 = popt_secondary

            if (plot):
                plt.figure(figsize=(8, 6))
                plt.plot(channels, counts, label="Raw Data", color="blue", marker='_')
                plt.plot(channels_main, gaussian(channels_main, *popt_main), label="Main Gaussian Fit", color="red", linestyle="--")
                plt.plot(channels_secondary, double_gaussian(channels_secondary, *popt_secondary), label="Secondary Gaussian Fits", color="green", linestyle="--")

                plt.xlim(0, 1250)
                plt.xlabel("Channel Number")
                plt.ylabel("Counts")
                plt.title(f"Gaussian Fits for {source}")
                plt.legend()
                plt.grid(True)
                plt.show()
            
            res.loc["Am241"] = [round(mu_main, 3), round(sigma_main, 3), round(amplitude_main, 3), round(count_rate_main, 3)]
            res.loc["Am241_secondary"] = [round(mu1, 3), round(sigma1, 3), round(amp1, 3), round(count_rate1, 3)]
            res.loc["Am241_third"] = [round(mu2, 3), round(sigma2, 3), round(amp2, 3), round(count_rate2, 3)]

        elif source == "Ba133":
            roi = roi_values["Ba133"]
            initial_params = initial_params_dict["Ba133"]
            popt, pcov, channels_roi, counts_roi, count_rate = fit_gaussian_spectrum(
                channels, counts, roi, initial_params, bounds_single, measurement_time
            )
            mu, sigma, amplitude = popt

            if (plot):
                plt.figure(figsize=(8, 6))
                plt.plot(channels, counts, label="Raw Data", color="blue", marker='_')
                plt.plot(
                    channels_roi,
                    gaussian(channels_roi, *popt),
                    label="Gaussian Fit",
                    color="red",
                    linestyle="--",
                )

                plt.xlim(500, 700)
                plt.xlabel("Channel Number")
                plt.ylabel("Counts")
                plt.title(f"Gaussian Fit for {source}")
                plt.legend()
                plt.grid(True)
                plt.show()
            
            res.loc["Ba133"] = [round(mu, 3), round(sigma, 3), round(amplitude, 3), round(count_rate, 3)]

    print(res)
    return res

def calibration_plot(res):
    known_energies = {
        "Am241": 59.5409,
        "Am241_secondary": 26.3446 ,
        "Am241_third": 33.1963,
        "Ba133": 53.1622,
        
    }

    mu_values = {
        "Am241": res["Mean(mu)"]["Am241"],
        "Am241_secondary": res["Mean(mu)"]["Am241_secondary"],
        "Am241_third": res["Mean(mu)"]["Am241_third"],
        "Ba133": res["Mean(mu)"]["Ba133"]
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
    plt.title("Energy Calibration Curve for CDTE Detector")
    plt.legend()
    plt.grid(True)
    plt.show()


    example_channel = 500
    print(f"Energy at channel {example_channel}: {channel_to_energy(example_channel, a, b):.2f} keV")


def resolution_plot(res):
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Am241_secondary": 26.3446 ,
        "Am241_third": 33.1963,
        "Ba133": 53.1622,
        
    }

    # Standard deviations obtained from Gaussian fitting
    sigma_values = {
        "Am241": res["Standard deviation"]["Am241"],
        "Am241_secondary": res["Standard deviation"]["Am241_secondary"],
        "Am241_third": res["Standard deviation"]["Am241_third"],
        "Ba133": res["Standard deviation"]["Ba133"],
    }

    # Calibration constants from linear fit
    a, b = 1.936, 10.315  
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
    energy_range = np.linspace(min(energies), max(energies), 500)  # 500 points for smooth curve
    smooth_resolution = np.sqrt(resolution_model(energy_range, *popt))

    # Plotting the energy resolution vs. energy
    plt.figure(figsize=(8, 6))
    plt.scatter(energies, np.sqrt(resolutions_squared), color='blue', label="Resolution Data")
    plt.plot(energy_range, smooth_resolution, 'r--', label="Fitted Resolution Model")  # Smooth fitted curve
    plt.xlabel("Energy (keV)")
    plt.ylabel("Energy Resolution (R)")
    plt.title("Energy Resolution vs. Energy for CdTe")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display fitted model parameters
    print(f"Fitted Resolution Model: R^2 = {a_fit:.4f} * E^-2 + {b_fit:.4f} * E^-1 + {c_fit:.4f}")

def efficiency_plot(res):
    # Known energies in keV for each source
    known_energies = {
        "Am241": 59.5409,
        "Am241_secondary": 26.3446 ,
        "Am241_third": 33.1963,
        "Ba133": 53.1622,   
    }

    # Source activities for each source (in Bq, or disintegrations per second)
    source_activities = {
        "Am241": 474340,
        "Am241_secondary": 474340,
        "Am241_third": 474340,
        "Ba133": 19938.19
    }

    # Count rates (counts per second) for each source from gaussian fitting
    count_rates = {
        "Am241": res["Count rate"]["Am241"],
        "Am241_secondary": res["Count rate"]["Am241_secondary"],
        "Am241_third": res["Count rate"]["Am241_third"],
        "Ba133": res["Count rate"]["Ba133"],
    }

    # Geometry factor for intrinsic efficiency calculation
    geometry_factor = 0.022

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
    plt.title("Intrinsic Efficiency vs. Energy for CdTe")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nFitted Efficiency Model: ln(ε) = {a_fit:.4f} + {b_fit:.4f} * ln(E) + {c_fit:.4f} * (ln(E))^2")

def off_axis_plot():
    # Define the file paths for the angle measurements
    file_paths = {
        "0deg": 'labdata/CDTE/CDTE_Detector_Am241_0deg_600sec.mca',
        "30deg": 'labdata/CDTE/CDTE_Detector_Am241_30deg_600sec.mca',
        "60deg": 'labdata/CDTE/CDTE_Detector_Am241_60deg_600sec.mca',
        "90deg": 'labdata/CDTE/CDTE_Detector_Am241_90deg_600sec.mca'
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
    plt.title("Counts vs. Channel for Different Angles in CdTe Detector")
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
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for CdTe Detector")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting intrinsic efficiency as a function of angle
    plt.figure(figsize=(7, 6))
    plt.plot(angles, intrinsic_efficiencies, marker='o', linestyle='--', color='green', label="Intrinsic Efficiency")
    plt.xlabel("Off-Axis Angle (degrees)")
    plt.ylabel("Efficiency / Geometric Factor")
    plt.title("Efficiency and Geometric Factor vs. Off-Axis Angle for CdTe Detector")
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