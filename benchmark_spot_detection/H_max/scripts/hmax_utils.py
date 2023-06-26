import numpy as np
import pandas as pd
from skimage.feature import blob_log
import scipy.optimize as opt
from skimage.morphology import extrema
from skimage.morphology import disk


def gauss_2d(xy:tuple, amplitude, x0, y0, sigma_xy, offset):
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

def gauss_single_spot(image: np.ndarray, c_coord: float, r_coord: float, crop_size=4,EPS = 1e-4) -> tuple:
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
        popt, pcov = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        #print('Runtime')
        return r_coord, c_coord, 0,0

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    sdx = sd[1]
    sdy = sd[2]

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return r_coord, c_coord, 0,0

    return y0, x0, sdx,sdy


def normalize(im:np.array)-> np.array:
    """Function to normalize the images by substracting the mean and dividing by the standard deviation

    Args:
        im (np.array): raw_image

    Returns:
        np.array: the normalized image
    """
    new_im = im.copy()
    #new_im = new_im.flatten()

    mean = np.mean(new_im)
    sd = np.std(new_im)
    new_im = ((new_im - mean) /sd) + 10000 # addition of 10000 to avoid having negative values

    return new_im

def get_loc(im:np.array,frame:int,mins:float,maxs:float,thresh:float,nums:int=10 )-> pd.DataFrame:

    """Function to return localizations from a laptrack detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details. Defaults to 10.

    Returns:
        pd.DataFrame: dataframe of all the localizations (gaussian fitted)
    """
    
    ima = im[frame].copy()
    ima = normalize(ima)
    
    df = lap(ima,mins=mins,maxs=maxs,nums=nums,thresh=thresh)
    x_loc =[]
    y_loc =[]
    for i in df.iloc:
        y,x,*_ = gauss_single_spot(ima,i.x,i.y)
        x_loc.append(x)
        y_loc.append(y)

    df['x'] = x_loc
    df['y'] = y_loc

    return df

def lap(im:np.array,mins:float,maxs:float,thresh:float,nums:int=10) -> pd.DataFrame: 
    """Function to compute laptrack spot detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details; Default to
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details

    Returns:
        pd.DataFrame: df of all the detections
    """
    images = im

    _spots = blob_log(images.astype(float), min_sigma=mins, max_sigma=maxs, num_sigma=nums,threshold_rel=thresh)
    
    rad = [np.sqrt(2*_spots[x][-1]) for x in range(len(_spots))]
    df = pd.DataFrame(_spots, columns=["y", "x", "sigma"])
    
    df['radius'] = rad
    
    df = df[df.radius > 1.2]
    
    df.reset_index(drop=True,inplace=True)
    return df

def hmax_detection(raw_im:np.array,frame:int,sd:float,n:int = 2,thresh:float = 0.5) -> pd.DataFrame:
    """_summary_

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        sd (float): the sd of the peak intensity (threshold of segmentation)
        n (int, optional): how much brighter than the sd of the whole image to threshold
        thresh (float, optional): threshold for the gaussian fitting filter. Filter on the standard deviation of the fit (based on the covariance of the parameters) Defaults to 0.5.

    Returns:
        pd.DataFrame: Dataframe of sub-pixel localizations of the detected spots
    """
    
    #detect the spots
    im_mask = extrema.h_maxima(raw_im[frame],n*sd)

    # extract the points and fit gaussian

    y,x = np.nonzero(im_mask)  # coordinates of every ones

    x_s = []
    y_s = []
    sdx_fit = []
    sdy_fit = []

    for i,j in zip (x,y):
        y0,x0,sd_x,sd_y = gauss_single_spot(raw_im[frame],i,j)
        x_s.append(x0)
        y_s.append(y0)
        sdx_fit.append(sd_x)
        sdy_fit.append(sd_y)
        
    # create a dataframe with sub pixel localization

    df_loc = pd.DataFrame([x_s,y_s,sdx_fit,sdy_fit]).T
    df_loc.rename(columns={0:'x',1:'y',2:'sd_fit_x',3:'sd_fit_y'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)
    df_loc
    
    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc[(df_loc.sd_fit_x.values < thresh) | (df_loc.sd_fit_y.values < thresh) ]
    df_loc_filtered

    return df_loc_filtered

def heatmap_detection(raw_im:np.array,frame:int,df:pd.DataFrame,name:str)-> tuple:
    """Create a heatmap to compute the threshold for h-max detection

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        df (pd.DataFrame): detections (x,y) on the raw image to be able to compute the intensity profiles of the detected spots
        name (str): either 'med' or 'sd'. Whether you want to display the median pixel intensity value around the spots or the sd. In any case it returns the values of the 2

    Returns:
        tuple: heatmap: a n-dimentional array (shape of the image) with either the median pixel intensity value for each bins or the sd of each bin, the median pixel intensity value for all detected spots provided, the sd of the intensity of every provided spots
    """
    # create image with extended boarders to be able to take bbox

    im = np.pad(raw_im[frame],2)

    # create the same for the mask

    im_mask = np.pad(np.ones_like(raw_im[frame]),2)

    spot = []
    for i in df.iloc:
        x,y = int(i.x+2),int(i.y+2)
        patch = im[x-2:x+3,y-2:y+3]
        patch_mask = im_mask[x-2:x+3,y-2:y+3]*disk(2)  # get only a disk in the bbox 
        spot.append(patch[patch_mask].ravel()) # get a 1d list of all bbox 

    med = np.median(spot,axis=1) # median of bbox where there are spots
    sd = np.std(spot,axis=1) # sd of bbox where there are spots

    df['med'] = med
    df['sd'] = sd

    df_heat = df[['x','y','med','sd']].copy(deep=True)

    try:
        df_heat['x'] = np.array(df_heat['x'].values//32)
        df_heat['y'] = np.array(df_heat['y'].values//32) # bin the image to categories to be able to see better the spots 

    except RuntimeWarning:
        df_heat['y'] = [0]*len(df_heat)
        df_heat['x'] = [0]*len(df_heat)


    heatmap = []
    for i in range(int(max(df_heat.y.values))):
        list_heat_row =[]
        for j in range(int(max(df_heat.x.values))):
            try:
                list_heat_row.append(np.mean(df_heat[(df_heat.x == j) & (df_heat.y == i)][name].values)) # average the region binned (row by row)
            except RuntimeWarning:
                list_heat_row.append(0)
                
        heatmap.append(list_heat_row)

    heatmap = np.array(heatmap)

    return heatmap,sd,med

