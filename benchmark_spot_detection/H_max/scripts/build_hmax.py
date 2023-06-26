import numpy as np
import pandas as pd
from hmax_utils import gauss_single_spot
from skimage.morphology import extrema
from tifffile import imread 
import re


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
    raw_im = np.expand_dims(raw_im,axis=0)
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
    if frame == 0:
        df_loc['frame'] = [frame] * len(df_loc)
    else:
        df_loc['frame'] = [-1] * len(df_loc)
    df_loc
    
    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc[(df_loc.sd_fit_x.values < thresh) | (df_loc.sd_fit_y.values < thresh) ]
    df_loc_filtered

    return df_loc_filtered

files = pd.read_csv(snakemake.input[0],header=None)# type: ignore
files = files[files.columns[0]].values

path_img = '/tungstenfs/scratch/ggiorget/nessim/spt/clean/'

sd = pd.read_csv(snakemake.input[1])# type: ignore


def build_hmax(files:list,sd:pd.DataFrame,n:int,thresh:float,path_img:str)-> pd.DataFrame:
    df_condition = pd.DataFrame()
    for file in files:

        im = imread(path_img+'/'+file)

        sd_img = sd[sd.image == file]
        sd_img = int(sd_img.sd.values[0])

        for frame in [0]:
            df_loc = hmax_detection(im,frame,sd_img,n,thresh)
            df_loc['raw_image'] = [file]*len(df_loc)
            df_loc['image'] = [file]*len(df_loc)
            df_condition = pd.concat([df_condition,df_loc])
    
    return df_condition


df_condition = build_hmax(files,sd,int(snakemake.wildcards.n),float(snakemake.wildcards.threshold_fit),snakemake.input[2])# type: ignore
df_condition['n'] = [int(snakemake.wildcards.n)]*len(df_condition)# type: ignore
df_condition['threshold_fit'] = [float(snakemake.wildcards.threshold_fit)]*len(df_condition)# type: ignore

df_condition.to_csv(snakemake.output[0]) # type:ignore
