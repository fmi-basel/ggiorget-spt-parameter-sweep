import pandas as pd
from laptrack import LapTrack
import re
import numpy as np


files = pd.read_csv(snakemake.input[2],header=None)# type: ignore
files = files[files.columns[0]].values


num = [x for x in files]
num = sorted(num)


def truth_table_sweep(df_exp,df_gt):
    
    ## building dataframe with 2 images to match
    
    df_assign = df_exp

    if df_exp.empty:
        return 0,0,0,0,0
    
    df_assign['frame'] = df_assign['frame'].values[0]+1 # to differentiate between ground truth and experimental data 
    df_assign = pd.concat([df_assign,df_gt])
    df_assign.dropna(axis=1,inplace=True)
    df_assign = df_assign[['x','y','frame']]
    
    # build the tracks (i.e the matched points)
    
    lt = LapTrack(track_cost_cutoff=2**2)# type: ignore 
    track_df, _, _ = lt.predict_dataframe(df_assign, ["y", "x"], only_coordinate_cols=False,validate_frame=False)
    track_df = track_df.reset_index()
    track_df = track_df[['x','y','frame','track_id']]
    
    # # get the duplicates in the track id (the matched points) and the unique unmatched
    # seen = set()
    # dupes = []
    
    # for x in track_df.track_id.values:
    #     if x in seen:
    #         dupes.append(x)
    #     else:
    #         seen.add(x)

    # unique = [x for x in range(len(track_df.index)) if x not in dupes]
    
    # # filter the dataframe for the points that were matched
    # match = track_df[track_df.track_id == dupes[0]]

    # for i in dupes:
    #     match = pd.concat([match,track_df[track_df.track_id == i]])

    # # filter for the points that weren't matched

    # un = track_df[track_df.track_id == unique[0]]

    # for j in unique:
    #     un = pd.concat([un,track_df[track_df.track_id == j]])

    # un = un.iloc[1:]
    u, c = np.unique(track_df.track_id.values, return_counts=True)
    dup = u[c > 1]
    uniq = u[c == 1]
    track_m = track_df[track_df.track_id.isin(dup)]
    track_u = track_df[track_df.track_id.isin(uniq)]

    false_positive = len(track_u[track_u.frame == 1]) # the unmatched points that come from the laptrack dataframe
    true_positive = round(len(track_m)/2) # the number of matched points divided by 2 (because 2 points per match)
    false_negative = len(track_u[track_u.frame == 0]) # The number of unmatched points coming from the ground truth dataframe
    true_negative = 1 #no way of knowing this one ..

    # if true_positive == 0:
    #     return 0,0,0,0,0
    
    precision = true_positive/(true_positive+false_positive)
    negative_predicted_value = true_negative/(true_negative+false_negative)
    sensitivity = true_positive/(true_positive+false_negative)
    specificity = true_negative/(true_negative+false_positive)
    accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)

    
    return precision,negative_predicted_value,sensitivity,specificity,accuracy


def build_condition(df_exp,num,df_gt):
    df_truth = []
    for i in num: #loop over images
        # for j in [0,-1]: #loop over frames
        test_astro = df_exp[df_exp.image == i]
        test_astro = test_astro[test_astro.frame == 0]

        test_gt = df_gt[df_gt.image == i]
        test_gt = test_gt[test_gt.frame == 0]              
        precision,negative_predicted_value,sensitivity,specificity,accuracy = truth_table_sweep(test_astro,test_gt)
        df = pd.DataFrame([precision,negative_predicted_value,sensitivity,specificity,accuracy])
        df_truth.append(df)
    return df_truth 

df_exp = pd.read_csv(snakemake.input[1])# type: ignore 
df_gt = pd.read_csv(snakemake.input[0])# type: ignore 

l = build_condition(df_exp,num,df_gt)
l = pd.DataFrame(np.array(l).reshape(10,5))
l.rename(columns={0:'precision',1:'negative_predicted_value',2:'sensitivity',3:'specificity',4:'accuracy'},inplace=True)
l.to_csv(snakemake.output[0])# type: ignore 

