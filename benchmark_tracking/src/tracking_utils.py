import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from laptrack import LapTrack

def open_gt(path_to_gt: str)-> pd.DataFrame:
    """
    :param path_to_gt: path to the ground truth file
    :return: a pandas dataframe with the ground truth
    """
    tracks = pd.read_csv(path_to_gt)
    # tracks = tracks.iloc[3:,:10]
    tracks.columns = ['b','bla','blas','track_id','fdx','x', 'y', 'z','intensity','frame','radius']
    tracks = tracks.drop(['b','bla','blas','fdx','intensity', 'radius'], axis=1)
    tracks.x = tracks.x.astype(float)
    tracks.y = tracks.y.astype(float)
    tracks.frame = tracks.frame.astype(int)
    tracks.track_id = tracks.track_id.astype(int)
    tracks.reset_index(drop=True, inplace=True)

    return tracks


def compute_frame_rate(time_btw_frame:int,max_frame:int) -> list:
    """
    :param time_btw_frame: time between two frames in s
    :param max_frame: maximum number of frames
    :return: a list of list of frame numbers
    """
    frame_rate = time_btw_frame//2

    seq = []

    comb_0 = np.arange(start=0,stop=max_frame,step=frame_rate)

    seq.append(comb_0)

    for i in range(1,comb_0[1]):
        seq.append(np.arange(start=i,stop=max_frame,step=frame_rate))

    return seq
    
def filter_df_based_on_frame(df:pd.DataFrame,seq:list)-> list:
    """
    :param df: pandas dataframe
    :param seq: list of list of frame numbers
    :return: a list of pandas dataframe
    """
    df_filtered = []
    for i in range(len(seq)):
        df_temp = df.loc[df.frame.isin(seq[i])]
        real_frames = list(df_temp['frame'].values)
        df_temp.rename(columns={'frame':'frame_real'},inplace=True)
        df_temp['frame'] = [list(seq[i]).index(df_temp.loc[j,'frame_real']) for j in df_temp.index]
        df_filtered.append(df_temp)
    return df_filtered

def create_filtered_image(im:np.array,seq:list)-> list:
    """
    :param im: numpy array of the image
    :param seq: list of list of frame numbers
    :return: a list of numpy array
    """
    im_filtered = [im[seq[i],:,:] for i in range(len(seq))]
    return im_filtered

def track(df:pd.DataFrame,track_cost_cutoff:int,gap_closing_max_frame_count:int,gap_closing_cost_cutoff:int,track_dist_metric:str = 'sqeuclidean') ->tuple:
    """Function to track spots in time

    Args:
        df (pd.DataFrame): the detection dataframe for all frames with x,y and frame as columns
        track_cost_cutoff (int, optional): The cutoff distance to consider when tracking (maximum distance to look for a spot to link). Defaults to 2.
        gap_closing_cost_cutoff (int, optional): . Defaults to 2.

    Returns:
        tuple: _description_
    """
    # Track using Lapt track (from emo notebook)
    lt = LapTrack(track_cost_cutoff=track_cost_cutoff**2,track_dist_metric=track_dist_metric,
                  gap_closing_cost_cutoff=gap_closing_cost_cutoff**2,gap_closing_max_frame_count=gap_closing_max_frame_count) # track_cost_cutoff and gap_closing_cutoff should be the squared maximum distance", 
    track_df, _, _ = lt.predict_dataframe(df, ["y", "x"], only_coordinate_cols=False,validate_frame=False)
    track_df = track_df.reset_index()
    track_df = track_df[['x','y','frame','track_id','frame_real']]

    # Find the repeated track_id (matched points)
    u, c = np.unique(track_df.track_id.values, return_counts=True)
    dup = u[c > 1]
    track_m = track_df[track_df.track_id.isin(dup)]
    
    return track_df,track_m

def compute_IoU(df1:pd.DataFrame,df2:pd.DataFrame) -> pd.DataFrame:
    """
    :param df1: track dataframe
    :param df2: ground_truth dataframe
    :return: a pandas dataframe with matched points and the associated IoU's
    """
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df1 = df1[['x','y','frame','track_id','frame_real']]
    df2 = df2[['x','y','frame','track_id','frame_real']]
    df = pd.merge(df1,df2,on=['x','y','frame'])
    df.rename(columns={'track_id_x':'track_id1','track_id_y':'track_id2'},inplace=True)

    n_common_spot = df.groupby(['track_id1','track_id2']).size().reset_index(name='count')

    n_common_spot['iou'] = n_common_spot.apply(lambda x: x['count']/((len(df1[df1.track_id==x['track_id1']])+len(df2[df2.track_id==x['track_id2']]))-x['count']),axis=1)

    return n_common_spot

