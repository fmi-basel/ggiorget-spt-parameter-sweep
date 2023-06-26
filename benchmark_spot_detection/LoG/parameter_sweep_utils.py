import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tifffile import imread
from skimage.feature import blob_log
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
import seaborn as sns
import os
import scipy.optimize as opt
import re
from laptrack import LapTrack


images = ['clean_9_',
 'clean_3_',
 'clean_5_',
 'clean_5_',
 'clean_7_',
 'clean_1_',
 'clean_3_',
 'clean_9_',
 'clean_1_',
 'clean_7_']

def get_files(path):
    # get only the files 
    docs = os.listdir(path)
    only_files = [f for f in docs if os.path.isfile(path+'/'+f)]
    files = []
    for file in only_files:
        for im in images:
            if im in file:
                files.append(file)

    files = list(set(files)) #remove repeated components

    return files # list of the files to open

# build astropy

def build_astropy(path,files,threshold):

    listdf = []
    frames = [0,-1] #first and last frames were tested 

    for file in files:
        im = imread(path+file)
        for frame in frames:
            df = starfind_sweep(im[frame],thresh=threshold).to_pandas()
            df['frame'] = frame
            df['image'] = int(re.findall(r'\d+',file)[2])
            listdf.append(df)

    astropy = pd.concat(listdf)
    return astropy

#build laptrack

def build_laptrack(path,files,mins,maxs,nums,thresh):

    listdf = []

    frames = [0,10]

    for file in files:
        im = imread(path+file)
        for frame in frames:
            df = get_loc(im[frame],mins=mins,maxs=maxs,nums=nums,thresh=thresh)
            df['frame'] = frame
            df['image'] = int(re.findall(r'\d+',file)[2])
            listdf.append(df)

    laptrack = pd.concat(listdf)
    
    return laptrack 

def starfind_sweep(data,thresh):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
    daofind = DAOStarFinder(fwhm=3.0, threshold=thresh*std)  
    sources = daofind(data - median)  
    for col in sources.colnames:  
        sources[col].info.format = '%.8g'  # for consistent table output
    return sources.to_pandas()

def lap_sweep(im,mins,maxs,nums,thresh):
    
    images = im

    spots = []
    _spots = blob_log(images.astype(float), min_sigma=mins, max_sigma=maxs, num_sigma=nums,threshold_rel=thresh)
    
    rad = [np.sqrt(2*_spots[x][-1]) for x in range(len(_spots))]
    df = pd.DataFrame(_spots, columns=["y", "x", "sigma"])
    
    df['radius'] = rad
    
    df = df[df.radius > 1.2]
    
    df.reset_index(drop=True,inplace=True)
    df = df.loc[:,['x','y','frame','image']]
    
    return df

def get_loc(im,mins,maxs,nums,thresh):
    
    n = lap_sweep(im,mins=mins,maxs=maxs,nums=nums,thresh=thresh)
    x_loc =[]
    y_loc =[]
    for i in n.iloc:
        y,x = gauss_single_spot(im,i.x,i.y)
        x_loc.append(x)
        y_loc.append(y)
        
    n['x'] = x_loc
    n['y'] = y_loc
        
    return n

# testing the dataframes

def truth_table_sweep(df_exp,df_gt):
    
    ## building dataframe with 2 images to match
    
    df_assign = df_exp
    df_assign['frame'] = df_assign['frame'].values[0]+1 # to differentiate between ground truth and experimental data 
    df_assign = pd.concat([df_assign,df_gt])
    df_assign.dropna(axis=1,inplace=True)
    df_assign = df_assign[['x','y','frame']]
    
    # build the tracks (i.e the matched points)
    
    lt = LapTrack(track_cost_cutoff=5**2)# type: ignore
    track_df, _, _ = lt.predict_dataframe(df_assign, ["y", "x"], only_coordinate_cols=False,validate_frame=False)
    track_df = track_df.reset_index()
    track_df = track_df[['x','y','frame','track_id']]
    
    # get the duplicates in the track id (the matched points) and the unique unmatched
    seen = set()
    dupes = []

    for x in track_df.track_id.values:
        if x in seen:
            dupes.append(x)
        else:
            seen.add(x)

    unique = [x for x in range(len(track_df.index)) if x not in dupes]

    # filter the dataframe for the points that were matched

    match = track_df[track_df.track_id == dupes[0]]

    for i in dupes:
        match = pd.concat([match,track_df[track_df.track_id == i]])

    # filter for the points that weren't matched

    un = track_df[track_df.track_id == unique[0]]

    for j in unique:
        un = pd.concat([un,track_df[track_df.track_id == j]])

    un = un.iloc[1:]
    
    false_positive = len(un[un.frame == 1]) # the unmatched points that come from the laptrack dataframe
    true_positive = round(len(match)/2) # the number of matched points divided by 2 (because 2 points per match)
    false_negative = len(un[un.frame == 0]) # The number of unmatched points coming from the ground truth dataframe
    true_negative = 1 #no way of knowing this one ...
    
    precision = true_positive/(true_positive+false_positive)
    negative_predicted_value = true_negative/(true_negative+false_negative)
    sensitivity = true_positive/(true_positive+false_negative)
    specificity = true_negative/(true_negative+false_positive)
    accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    
    return precision,negative_predicted_value,sensitivity,specificity,accuracy

def gauss_2d(xy, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
        )
    )
    return gauss

def gauss_single_spot(
    image: np.ndarray, c_coord: float, r_coord: float, crop_size=4
):
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max([int(np.round(r_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(c_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop 
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop 
    sigma = max(*crop.shape) * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(np.max(crop) / 2, np.min(crop))  # Height of gaussian, maximum value
    initial_guess = [amplitude_max, x0, y0, sigma, 0]

    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, _ = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
    except RuntimeError:
        return r_coord, c_coord

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return r_coord, c_coord

    return y0, x0

EPS = 1e-4