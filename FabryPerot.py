from skimage.draw import line
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import statistics
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.signal import savgol_filter
import csv





def scan_top_down(S1, S2, scan_bin, max_pixel, img, x_scan, y_scan, L_peak):
    for k in np.arange(S1, S2, scan_bin):

        rr, cc = line(y_scan, x_scan, max_pixel, k)
        intensities = img[rr, cc]
        peaks, _ = find_peaks(intensities, prominence=11, height=15, distance=50)
        distance = 0
        if len(peaks)>=1:
            distance += np.sqrt((rr[peaks[0]] - y_scan)**2 + (cc[peaks[0]]-x_scan)**2)
            L_peak.append(distance)
    return L_peak

def scan_left_right(S1, S2, scan_bin, max_pixel, img, x_scan, y_scan, L_peak):
    for k in np.arange(S1, S2, scan_bin):

        rr, cc = line(y_scan, x_scan, k, max_pixel)
        intensities = img[rr, cc]
        peaks, _ = find_peaks(intensities, prominence=11, height=15, distance=50)
        distance = 0
        if len(peaks)>=1:
            distance += np.sqrt((rr[peaks[0]] - y_scan)**2 + (cc[peaks[0]]-x_scan)**2)
            L_peak.append(distance)
    return L_peak



#X500 750 400 650

def center_finder(image_file, X_range=(600, 650), Y_range=(500, 550), bin_range=2, scan_range=(500, 600),scan_range_top=(500, 600), scan_bin=8, plotting=False):
    """Retourne les coordonnées du centre des cerlces d'interference concentriques"""
    L_coord = []
    L_ecart = []
    X1, X2 = X_range
    Y1, Y2 = Y_range
    S1, S2 = scan_range
    S1_top, S2_top = scan_range_top

    rows = Y2-Y1
    columns = X2-X1
    Lzz = np.zeros((rows, columns), dtype=int)


    #Importation image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Boucle principale
    for x_scan in np.arange(X1, X2, bin_range):
        for y_scan in np.arange(Y1, Y2, bin_range):
            L_peak = []

            L_peak = scan_top_down(S1, S2, scan_bin, 0, img, x_scan, y_scan, L_peak)
            L_peak = scan_top_down(S1, S2, scan_bin, 1020, img, x_scan, y_scan, L_peak)
            L_peak = scan_left_right(S1, S2, scan_bin, 0, img, x_scan, y_scan, L_peak)
            L_peak = scan_left_right(S1, S2, scan_bin, 1270, img, x_scan, y_scan, L_peak)

            L_ecart.append(np.std(L_peak))

            #Liste pour l'affichage
            L_coord.append([x_scan, y_scan, np.std(L_peak)])
            Lzz[y_scan-Y1, x_scan-X1] = np.std(L_peak)

    #Coordonnées du centre 

    L_sort = np.sort(np.array(L_ecart))


    X_center = L_coord[L_ecart.index(min(L_ecart))][0]
    Y_center = L_coord[L_ecart.index(min(L_ecart))][1]


    if plotting == True:
        plt.pcolormesh(Lzz, cmap='inferno')
        plt.colorbar()
        plt.xlabel('Lx')
        plt.ylabel('Ly')

    return (X_center, Y_center)








def peak_finder(image_file, prom, h, d, X_center, Y_center):
    """Trouve les pics correspondants à la position des anneaux"""

    #Importation image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = [(X_center, Y_center, j, 1270) for j in range(0, 1020)] + [(X_center, Y_center, j, 0) for j in range(0, 1020)] + [(X_center, Y_center, 1020, j) for j in range(0, 1270)] + [(X_center, Y_center, 0, j) for j in range(0, 1270)]

    Lpeaks = []
    for l in lines:
        rr, cc = line(X_center, Y_center, *l[2:])
        intensities = img[rr, cc]
        peaks, _ = find_peaks(intensities, prominence=prom, height=h, distance=d)
        for peak in peaks:
            Lpeaks.append(np.sqrt((rr[peak] - X_center)**2 + (cc[peak]-Y_center)**2))

    L_distance = np.zeros(int(max(Lpeaks))+1)
    for i in Lpeaks:
        L_distance[int(i)] += 1

    return L_distance


Peak_start = 190



def plotting_rings(image_file, X_cal, Y_cal):


    PROM_init = 3 #3
    D_init = 10 #10

    PROM = 23 #75
    H = 5 
    D = 10
    W = 1
    decalage = -150
    peak_index = 0
    A=0.45


    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    L_d = peak_finder(image_file, prom=PROM_init, d=D_init, h=H, X_center=Y_cal, Y_center=X_cal)
    peaks, _ = find_peaks(L_d[Peak_start:], prominence=PROM, height=H, distance=D, width=W)
    height_func = -A*np.linspace(1, len(L_d[Peak_start:]), len(L_d[Peak_start:])) + L_d[Peak_start:][peaks[peak_index]] + peaks[peak_index] + decalage
    peaks, _ = find_peaks(L_d[Peak_start:], prominence=PROM, height=height_func, distance=D, width=W)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='inferno')
    for i in peaks:
        circle = Circle((X_cal, Y_cal), radius=Peak_start+i, color='red', fill=False)
        ax.add_patch(circle)
    ax.set_xlabel("Lx (pixels)")
    ax.set_ylabel("Ly (pixels)")
    return peaks, L_d, height_func





def Lorentz(x, amplitude, center, width):
    return amplitude * (width**2 / ((x - center)**2 + width**2))

def make_sum(num_funcs):
    def Lorentz_sum(x, *params):
        res = 0
        for i in range(num_funcs):
            res += Lorentz(x, L_d[Peak_start:][peaks][i], peaks[i], params[i])
        return res
    return Lorentz_sum







def peak_measurement(X_cal, Y_cal, peaks, L_d, sigmas, height_func, title):
    """Retourne la valeur des pics et leurs incertitudes respectives. Affiche également ces derniers et le fit lorentzien"""

    num_funcs = len(peaks)
    amplitude = np.ones(num_funcs)*50
    center = np.array(peaks)
    width = np.ones(num_funcs)

    p0 = width
    Ld_lim = L_d[Peak_start:]



    # Fit the data
    popt, _ = curve_fit(make_sum(num_funcs), np.linspace(0, len(Ld_lim), len(Ld_lim)), Ld_lim, p0,  maxfev=1000000)
    x_fit = np.linspace(0, len(Ld_lim), len(Ld_lim))
    y_fit = make_sum(num_funcs)(x_fit, *popt)



    fig2, ax2 = plt.subplots()
    ax2.plot(np.linspace(0, len(Ld_lim), len(Ld_lim)),height_func, color="black", label="limite", linestyle="--")
    ax2.plot(x_fit,y_fit, color="b", label="ajustement")
    ax2.scatter(peaks, Ld_lim[peaks], color="r", label="pics")
    ax2.plot(np.linspace(0, len(Ld_lim), len(Ld_lim)), Ld_lim, label="données : X={}, Y={}".format(X_cal, Y_cal), color="g")
    ax2.set_xlabel("distance du pic (pixels)")
    ax2.set_ylabel("nombre de pics")
    ax2.set_title(title)
    ax2.legend()

    peaks_inc = sigmas*np.array(popt)/2


    

    return peaks, peaks_inc, Ld_lim





def data_writing(peaks, peaks_inc, ordre, filename, filename2, B, peak_width_base, TYPE="3", first=True):
    L_lambda = ["B (T)"]
    L_lambda_inc = []
    L_peaks = [B]
    L_peaks_inc = []

    for i in range(3*5):
        L_lambda.append("lambda" + str(i))
        L_lambda_inc.append("lambda_inc" + str(i))



    if TYPE == "1":
        for i in range(ordre):

            width = (peaks_inc[i]-peak_width_base[i])/2
            L_peaks.append(L_peaks[i]-width)
            L_peaks.append(L_peaks[i])
            L_peaks.append(L_peaks[i]+width)

            L_peaks_inc.append(peaks_inc[i])
            L_peaks_inc.append(peaks_inc[i])
            L_peaks_inc.append(peaks_inc[i])





    elif TYPE == "3":
        for i in range(3*ordre):  #add -1 if you have only two peaks
            L_peaks.append(peaks[i]+Peak_start)
            L_peaks_inc.append(peaks_inc[i])

        # L_peaks.append(L_peaks[-1]-L_peaks[-2]+L_peaks[-1])
        # L_peaks_inc.append(L_peaks_inc[-2])


    L_col =  L_lambda + L_lambda_inc
    L_data = L_peaks + L_peaks_inc

    csvfile = open(filename, 'a', newline='')
    csvwriter = csv.writer(csvfile)

    if first==True:    
        csvwriter.writerow(L_lambda)
    csvwriter.writerow(L_peaks)
    csvfile.close()

    csvfile = open(filename2, 'a', newline='')
    csvwriter = csv.writer(csvfile)

    if first==True:    
        csvwriter.writerow(L_lambda_inc)
    csvwriter.writerow(L_peaks_inc)
    csvfile.close()















peak_inc = []

def data_print(image_file, X_cal=1, Y_cal=1, plotting=True):


    #615 675 490 550
    if X_cal ==1:
        X_cal, Y_cal = center_finder(image_file, X_range=(415, 875), Y_range=(290, 750), bin_range=1, scan_bin=128, plotting=True)
    print(X_cal, Y_cal)
    #X_cal, Y_cal =  644, 521


    #[2.70542531 2.08298788 1.7539036  1.82185597 1.68959836 1.50375354 1.55335043]
    global L_d
    global peaks
    global height_func
    X_cal, Y_cal = 646, 524
    peaks, L_d , height_func = plotting_rings(image_file, X_cal, Y_cal)


    peaks, peaks_inc, L_d = peak_measurement(X_cal, Y_cal, peaks, L_d, 1, height_func, image_file)

    # for i in range(len(peaks)):
    #     peaks[i] = peaks[i] + Peak_start

    peak_inc.append(peaks_inc[0])

    print(peaks, peaks_inc)
    return peaks, peaks_inc






#643 521
peaks, peaks_inc = data_print("Zeeman rouge (0mT, 0A).tif", plotting=True)





L_I = [0, 0.5, 1, 1.5, 2, 2.5, 3]

# fig4, ax3 = plt.subplots()
# ax3.scatter(np.linspace(0, len(peak_incc), len(peak_incc)), peak_incc)







peak_width_base = [1.83580725, 1.46995397, 1.81754718, 3.26142105, 2.0919558,  1.59689622]
#data_writing(peaks, peaks_inc, 6, "cadmium_red.csv", "cadmium_red_inc.csv", 0.5736, peak_width_base, TYPE="3", first=False)








# fig3, ax3 = plt.subplots()
# ax3.scatter(np.linspace(1, len(peaks), len(peaks)), peaks_inc)
# #ax3.errorbar(np.linspace(1, len(peaks), len(peaks)), peaks, yerr=peaks_inc)

plt.show()
