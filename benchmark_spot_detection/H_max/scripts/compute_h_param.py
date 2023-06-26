from hmax_utils import get_loc,heatmap_detection,lap
import numpy as np
from tifffile import imread
import pandas as pd

mins_d = 2.111
maxs_d = 3.737
thresh_d = 0.884

mins_c = 1.837
maxs_c = 3.947
thresh_c = 0.837

mins_n = 1.974
maxs_n = 3.000
thresh_n = 0.884

files = pd.read_csv(snakemake.input[0],header=None)# type: ignore
files = files[files.columns[0]].values

if 'noisy' in files[0]:
    mins,maxs,threshold = mins_n,maxs_n,thresh_n
    c = 'noisy'

elif 'clean' in files[0]:
    mins,maxs,threshold = mins_c,maxs_c,thresh_c
    c = 'clean'

elif 'denoised' in files[0]:
    mins,maxs,threshold = mins_d,maxs_d,thresh_d
    c = 'denoised'
else:
    mins,maxs,threshold = mins_d,maxs_d,thresh_d
    c = 'denoised'

#path_img = '/tungstenfs/scratch/ggiorget/nessim/spt/' + c + '/'

def compute_h_param(file:list,mins:float,maxs:float,thresh:float,path_img:str) -> float:
    mean_sd_list = []

    for f in file:
        im = imread(path_img+'/'+f)
        im = np.expand_dims(im,axis=0)

        for frame in [0]:

            # Compute LoG with very high threhsold 

            df = lap(im[frame],mins,maxs,thresh)

            #compute the sd of the detected spots

            _,sd,_ = heatmap_detection(im,frame=frame,df=df,name='sd')

            # compute the mean sd across the image 

            mean_sd = np.mean(sd)

            mean_sd_list.append(mean_sd)

    return mean_sd_list


mean_sd_list = compute_h_param(file = files,mins=mins,maxs=maxs,thresh=threshold,path_img=snakemake.input[1])# type: ignore


df = pd.DataFrame(mean_sd_list)
df.rename(columns={0:'sd'},inplace=True)

df['frame'] = [0]*int((len(df)))

f = np.array([f for f in files]).flatten()
df['image'] = f

df.to_csv(snakemake.output[0]) #type: ignore 

# TO DO

# Find where the RuntimeWarning Mean of empty slice and invalid value encountered in double scaler comes from and implement a try excpet rule