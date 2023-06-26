
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from tifffile import imread
import pandas as pd
import re


def build_astropy(path,files,fwhm,threshold):

    listdf = []
    frames = [0,-1] #first and last frames were tested 

    for file in files:
        im = imread(path+"/"+file)
        for frame in frames:
            df = starfind_sweep(im[frame],fwhm,thresh=threshold)
            df['frame'] = frame
            df['image'] = int(re.findall(r'\d+',file)[2])
            df.rename(columns={'xcentroid':'x','ycentroid':'y'},inplace=True)
            listdf.append(df)

    astropy = pd.concat(listdf)
    return astropy

def starfind_sweep(data,fwhm,thresh):
    _, median, std = sigma_clipped_stats(data, sigma=3.0)  
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)  
    sources = daofind(data - median)  
    if sources is None:
        return pd.DataFrame()
    else:
        for col in sources.colnames:  
            sources[col].info.format = '%.8g'  # for consistent table output
    d = sources.to_pandas() 
    return d

files = pd.read_csv(snakemake.input[1],header=None)# type: ignore
files = files[files.columns[0]].values

f = build_astropy(snakemake.input[0],files,float(snakemake.wildcards.fwhm_astro),float(snakemake.wildcards.threshold_astropy))# type: ignore

f.to_csv(snakemake.output[0])# type: ignore

