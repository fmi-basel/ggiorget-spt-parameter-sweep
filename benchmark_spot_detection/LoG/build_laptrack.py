import pandas as pd
import numpy as np
from tifffile import imread
from skimage.feature import blob_log
import re
from parameter_sweep_utils import gauss_single_spot

def normalize(im):

    new_im = im.copy()
    mean = np.mean(new_im)
    sd = np.std(new_im)
    new_im = ((new_im - mean)/sd)+ 10000 # shift the distribution to avoid negative values of intensities

    return new_im

def lap_sweep(im,mins,maxs,nums,thresh):
    
    images = im
    _spots = blob_log(images.astype(float), min_sigma=mins, max_sigma=maxs, num_sigma=nums,threshold_rel=thresh)
    
    rad = [np.sqrt(2*_spots[x][-1]) for x in range(len(_spots))]
    df = pd.DataFrame(_spots, columns=["y", "x", "sigma"])
    
    df['radius'] = rad
    
    df = df[df.radius > 1.2]
    df.reset_index(drop=True,inplace=True)

    if len(df.index) > 4000:
        df = pd.DataFrame()
        
    return df


def get_loc(im,mins,maxs,nums,thresh):
    
    ima = normalize(im)

    n = lap_sweep(ima,mins=mins,maxs=maxs,nums=nums,thresh=thresh)

    x_loc =[]
    y_loc =[]

    for i in n.iloc:
        y,x = gauss_single_spot(im,i.x,i.y)
        x_loc.append(x)
        y_loc.append(y)
        
    n['x'] = x_loc
    n['y'] = y_loc
        
    return n

def build_laptrack(path,files,mins,maxs,nums,thresh):

    listdf = []

    frames = [0,-1]

    for file in files:
        im = imread(path+'/'+file)

        for frame in frames:
            df = get_loc(im[frame],mins=mins,maxs=maxs,nums=nums,thresh=thresh)
            df['frame'] = frame
            df['image'] = int(re.findall(r'\d+',file)[2])
            df = df.loc[:,['x','y','frame','image']]
            listdf.append(df)

    laptrack = pd.concat(listdf)
    
    return laptrack 

# files = pd.read_csv(snakemake.input[1],header=None)# type: ignore
# files = files[files.columns[0]].values

# laptrack = build_laptrack(snakemake.input[0],files,float(snakemake.wildcards.mins),float(snakemake.wildcards.maxs),snakemake.params[0],float(snakemake.wildcards.threshold_laptrack))# type: ignore

# laptrack.to_csv(snakemake.output[0])# type: ignore
