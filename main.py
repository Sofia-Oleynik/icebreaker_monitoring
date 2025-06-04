import os 
import tkinter as tk 
from tkinter import filedialog 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure 
from PIL import Image 
import skimage.measure as measure 
import matplotlib.patches as mpatches 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import matplotlib.patches as mpatches 
import zipfile 
import numpy as np 
np.random.seed(666) 
import pandas as pd 
from PIL import ImageTk, Image 
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout 
from keras.layers import Flatten, Dense 
from keras.models import Model, load_model 
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping 
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import uniform_filter 
from scipy.ndimage.measurements import variance 
def normalize(sig_nought, inc_angle): 
    """ 
    :type sig_nought: np.ndarray(np.float)  
    :type inc_angle: float 
    """ 
    sig_nought_n = (sig_nought + 0.766 * inc_angle - 31.638) / 2 
 
    return sig_nought_n 
def lee_filter(img, size): 
    img_mean = uniform_filter(img, (size, size)) 
    img_sqr_mean = uniform_filter(img**2, (size, size)) 
    img_variance = img_sqr_mean - img_mean**2 
 
    overall_variance = variance(img) 
 
    img_weights = img_variance / (img_variance + overall_variance) 
    img_output = img_mean + img_weights * (img - img_mean) 
    return img_output 
def normalize_data(hh, hv, inc_angle): 
    # normalize hh and hv using inc_angle (x_angle[i]) 
    hh = normalize(hh, inc_angle) 
    hv = normalize(hv, inc_angle) 
 
    # apply Lee filter for all bands 
    hh = lee_filter(hh, 20) 
    hv = lee_filter(hv, 20) 
 
    # total backscatter = hh + hv 
    b3 = hh + hv 
    # cross polarisation ratio = hv / (hh+hv) 
    c3 = hv / 2 * (hh + hv) 
 
# Rescale images between 0 and 1 for faster convergence rate 
hh = (hh - hh.min()) / (hh.max() - hh.min()) 
hv = (hv - hv.min()) / (hv.max() - hv.min()) 
b3 = (b3 - b3.min()) / (b3.max() - b3.min()) 
c3 = (c3 - np.nanmin(c3)) / (np.nanmax(c3) - np.nanmin(c3)) 
return hh, hv, b3, c3 
model = load_model("~\model_100epcs.hd5") 
path = r"D:/путь/" 
file = "S1A_IW_GRDH_1SDH_20210131T204712_20210131T204737_036387_044542_D668_output.zip" 
wdir = r"D:/путь/" + file[:-4] 
if not os.path.exists(wdir): 
os.makedirs(wdir) 
os.chdir(wdir) 
# Import bands 
hh = np.array(Image.open(file[:-11] + '_cal_ter_db_UTM_msk_Sigma0_HH_db.tif')) 
hv = np.array(Image.open(file[:-11] + '_cal_ter_db_UTM_msk_Sigma0_HV_db.tif')) 
inc_angle = np.array(Image.open(file[:-11] + '_cal_ter_db_UTM_msk_IncidenceAngle.tif')) 
# Also import the arrays containing the latitude and longtitude data. 
lat = np.array(Image.open(file[:-11] + '_cal_ter_db_UTM_msk_lat.tif')) 
lon = np.array(Image.open(file[:-11] + '_cal_ter_db_UTM_msk_lon.tif')) 
hh, hv, b3, c3 = normalize_data(hh, hv, inc_angle) 
def plot_axes(x, y, data, title): 
ax_p = axes[x, y].imshow(data, cmap="Greys_r") 
divider = make_axes_locatable(axes[x,y]) 
cax = divider.append_axes("right", size="5%", pad=0.08) 
fig.colorbar(ax_p, ax=axes[x,y], cax=cax) 
axes[x,y].set_title(title) 
fig, axes = plt.subplots(2,2, figsize=(20,10)) 
plot_axes(0,0, hh, 'HH') 
plot_axes(0,1, hv, 'HV') 
plot_axes(1,0, b3, 'Total backscatter (b3)') 
plot_axes(1,1, c3, 'Cross polarization ratio (c3)') 
plt.show() 
thresh = np.median(b3) + 5*np.std(b3) 
print(thresh) 
# Create array of zeros 
roi = np.zeros(np.shape(b3)) 
# Mark all potential objects with ones 
roi[b3>thresh] = 1 
plt.figure(figsize=(10,5)) 
plt.imshow(roi) 
plt.colorbar() 
plt.show() 
# Assign unique value for each region of same values 
roi_labeled = measure.label(roi, connectivity=2) 
print("Number of potential objects before removing small ones: " + 
str(np.max(roi_labeled))) 
# Count occurences of unique values 

unique, counts = np.unique(roi_labeled, return_counts=True) 
# Get index of all regions consisting of less then 5 pixels 
# np where returns a tuple of an np.array, [0] at end gets only the 
np.array 
idx = np.where(counts < 3)[0] 
# Assign 0 to all pixels in the roi-mask that are only small regions 
for i in idx: 
roi[roi_labeled == i] = 0 
# Assign unique value for each region of same values again, now without 
the small regions 
roi_labeled = measure.label(roi, connectivity=2) 
print("\nNumber of potential objects after removing small ones: " + 
str(np.max(roi_labeled)) + "\n") 
plt.figure(figsize=(10, 5)) 
plt.imshow(roi) 
plt.colorbar() 
plt.title("Plot of all popotential objects") 
plt.show() 
# Get bboxes of each region 
bboxes = [area.bbox for area in measure.regionprops(roi_labeled)] 
# Remove all bboxes which are at the border of the image (likely not fully 
within the image) 
bboxes = [bound for bound in bboxes if bound[0] > 0 and 
187 
bound[2] < np.shape(b3)[0] and 
bound[1] > 0 and 
bound[3] < np.shape(b3)[1]] 
print("\nNumber of potential objects after removing ones at the edges: " 
+ str(len(bboxes)) + "\n") 
geo_output = [] 
lat_max, lon_max = np.shape(lat) 
lat_max = lat_max-1 
lon_max = lon_max-1 
geo_output.append([lat[0,0], lon[0,0], "upper left corner", np.nan]) 
geo_output.append([lat[0,lon_max], lon[0,lon_max], "upper right 
corner", np.nan]) 
geo_output.append([lat[lat_max,lon_max], lon[lat_max,lon_max], "lower 
right orner", np.nan]) 
geo_output.append([lat[lat_max,0], lon[lat_max,0], "lower right corner", 
np.nan]) 
# Linewidth for Plot 
lw = 3 
fig, ax = plt.subplots(figsize=(10, 10)) 
ax.imshow(b3, cmap="Greys") 
for bbox in bboxes: 
# Calculate the extend of bbox 
height = bbox[2] - bbox[0] 
188 
189 
 
    width = bbox[3] - bbox[1] 
    # coordinates of upper left pexel of bbox (can be used for plotting) 
    ul = [bbox[1], bbox[0]] 
 
    # Calculate center of bbox 
    c_x = (bbox[0] + bbox[2]) / 2 
    c_y = (bbox[1] + bbox[3]) / 2 
    roi_center = [c_x, c_y] 
 
    # Define extend of 75*75 subset with center of bbox in the middle 
    # a = sub_roi 
    x1 = round(roi_center[0]) - 38 
    x2 = round(roi_center[0]) + 37 
    y1 = round(roi_center[1]) - 38 
    y2 = round(roi_center[1]) + 37 
 
    # If parts of the subset would be out of bounds of original image, 
place the subplot at the border of the image 
    if x1 < 0: 
        x1 = 0 
        x2 = 75 
    if y1 < 0: 
        y1 = 0 
        y2 = 75 
 
    if x2 > np.shape(b3)[0]: 
        x1 = np.shape(b3)[0] - 76 
        x2 = np.shape(b3)[0] - 1 

    if y2 > np.shape(b3)[1]: 
        y1 = np.shape(b3)[1] - 76 
        y2 = np.shape(b3)[1] - 1 
 
    sub_hh = hh[x1:x2, y1:y2] 
    sub_hv = hv[x1:x2, y1:y2] 
    sub_b3 = b3[x1:x2, y1:y2] 
    sub_c3 = c3[x1:x2, y1:y2] 
 
    sub_img = np.dstack((sub_hh, sub_hv, sub_b3, sub_c3)) 
    sub_img = np.expand_dims(sub_img, axis=0) 
 
    # feed sub_img into the model to predict 
    output = model.predict(sub_img) 
 
    # print(output) 
 
    if output > 0.05 and output < 0.3 : 
        # Ship! 
        geo_output.append([lat[int(roi_center[0]), int(roi_center[1])], 
                           lon[int(roi_center[0]), int(roi_center[1])], 
                           "Судно", output[0][0]]) 
        box = mpatches.Rectangle((ul[0], ul[1]), width, height, 
                                 edgecolor='tab:red', linewidth=lw, fill=None) 
    elif output > 0.4: 
        # Iceberg! 
        geo_output.append([lat[int(roi_center[0]), int(roi_center[1])], 
                           lon[int(roi_center[0]), int(roi_center[1])], 
                           "Айсберг", output[0][0]]) 

        box = mpatches.Rectangle((ul[0], ul[1]), width, height, 
                                 edgecolor='tab:blue', linewidth=lw, fill=None) 
    else: 
        # unidentified floating object 
        geo_output.append([lat[int(roi_center[0]), int(roi_center[1])], 
                           lon[int(roi_center[0]), int(roi_center[1])], 
                           "Неопознано", output[0][0]]) 
        box = mpatches.Rectangle((ul[0], ul[1]), width, height, 
                                 edgecolor='tab:purple', linewidth=lw, fill=None) 
 
    ax.add_patch(box) 
    ax.text(ul[0] - 10, ul[1] - 15, str(round(output[0][0], 2))) 
 
# Add legend to plot 
l1 = mpatches.Patch(edgecolor='tab:red', facecolor='None', lw=2, label='Судно') 
l2 = mpatches.Patch(edgecolor='tab:blue', facecolor='None', lw=2, label='Айсберг') 
l3 = mpatches.Patch(edgecolor='tab:purple', facecolor='None', lw=2, label='Не опознано') 
legend = ax.legend(handles=[l1, l2, l3], loc='upper right') 
 
legend.legendPatch.set_facecolor('darkgrey') 
 
plt.show() 
plt.savefig('plot.png') 
 
 
 
root = tk.Tk() 
root.title("Поиск айсбергов") 
file_label = tk.Label(root, text="Путь к файлу:") 
file_label.pack() 
file_entry = tk.Entry(root) 
file_entry.pack() 
execute_button = tk.Button(root, text="Выполнить", command=execute) 
execute_button.pack() 
output_label = tk.Label(root) 
output_label.pack() 
xml_button = tk.Button(root, text="Вывод в формат xml", command=save_xml) 
xml_button.pack() 
root.mainloop() 
root.mainloop() 
geo_output = pd.DataFrame(geo_output,columns=['lat', 'lon', 'object', 'probability']) 
geo_output.to_csv('~' + file[:-11] +  ".csv", sep="\t", index=False) 
geo_output.tail()