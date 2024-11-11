'''
This is the main script which is the compilation all the three detector (BGO,NaI,CdTe) scripts.
once you run this you get to select what detector and its value you need
'''
def get_det_input():
    det = input("Enter the detector (BGO/CDTE/NAI): ").upper()
    if det not in ["BGO", "CDTE", "NAI"]:
        print("Enter one of the given detectors")
        det = get_det_input()
    return det

# This function prompts the user to pick a detector and generate the outputs from the respetive detector 
def get_plot_input():
    det = int(input("Enter the plot number (counts[1]/curve[2]/calibration[3]/resolution[4]/efficiency[5]/offaxis[6]/all[7]): "))
    if det not in range(1, 8):
        print("Enter one of the given plots")
        det = get_plot_input()
    return det

det = get_det_input()

if det == "BGO":
    from BGO_Detector_Code import *
elif det == "CDTE":
    from CDTE_Detector_Code import *
else:
    from NAI_detector_code import *

# User is prompted to select what type of data they want to see
plot = get_plot_input()
if (plot == 1):
    count_plot()
elif (plot == 2):
    res = curve_plot()
elif (plot == 3):
    res = curve_plot(plot=False)
    calibration_plot(res)
elif (plot == 4):
    res = curve_plot(plot=False)
    resolution_plot(res)
elif (plot == 5):
    res = curve_plot(plot=False)
    efficiency_plot(res)
elif (plot == 6):
    off_axis_plot()
elif (plot == 7):
    all_plots()