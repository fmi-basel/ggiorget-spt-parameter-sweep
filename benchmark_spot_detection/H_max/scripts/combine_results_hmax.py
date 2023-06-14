import pandas as pd
import os

def combine_files(path):
    listdf = []
    for i in os.listdir(path):
        if 'parameter_sweep' in i:
            if 'hmax' in i:
                df = pd.read_csv(path+'/'+i)
                df['param'] = [i[21:-4]]*len(df.index)
                listdf.append(df)
    listdf = pd.concat(listdf)
    return listdf

list_df = combine_files(snakemake.input[0])# type: ignore 

list_df.to_csv(snakemake.output[0])# type: ignore 
