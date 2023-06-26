import pandas as pd
import numpy as np
import os
import nums_from_string as nfs
from more_itertools import consecutive_groups
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_param(df_heat,metric,n_top):
    df_heat['param'] = df_heat['track_cost_cutoff'].astype(str) + '_' + df_heat['gap_closing_cost_cutoff'].astype(str) + '_' + df_heat['gap_closing_max_frame_count'].astype(str)
    df_heat['ranking'] = df_heat[metric].rank(method='first',ascending=False)
    df_heat = df_heat.sort_values(by='ranking')

    # Get the n_top top parameters 

    best_params = [df_heat.iloc[0].param] #list of the best params

    for i in df_heat.iloc:
        if i.param not in best_params :
            best_params.append(i.param)
            if len(best_params) > n_top: # check that you only take n_top
                break                                                    

    df_best_params = pd.DataFrame()

    for l,i in enumerate(best_params):
        df = df_heat[df_heat.param == i].copy(deep=True)
        df.loc[:,'top'] = l+1
        df_best_params = pd.concat([df_best_params,df])

    df_best_params.reset_index(inplace=True,drop=True)

    return df_best_params

def get_tracks_params(path):
    list_df = pd.DataFrame()
    for i in os.listdir(path):
        if 'track_cost' in i:
            tracks = pd.read_csv(path+i)
            avg_track_len = np.mean([len(tracks[tracks.track_id==i]) for i in tracks.track_id.unique()])
            # print(max([len(tracks[tracks.track_id==i]) for i in tracks.track_id.unique()]))
            track_number = len(tracks.track_id.unique())
            track_df = pd.DataFrame(columns=['track_number','avg track length','param'])
            track_df['track_number'] = [track_number]
            track_df['avg track length'] = [avg_track_len]
            track_df['param'] = str(nfs.get_nums(i))
            list_df = pd.concat([list_df,track_df])
        else:
            continue
    
    return list_df

def get_gaps(df:pd.DataFrame) -> pd.DataFrame:
    """
    :param df: track dataframe
    :return: a pandas dataframe with gaps per track id
    """
    df = df.reset_index()
    df = df[['x','y','frame','track_id']]
    df['frame'] = df['frame'].astype(int)
    df = df.sort_values(by=['track_id','frame'])
    df['frame_diff'] = df['frame'].diff()
    df['frame_diff'] = df['frame_diff'].fillna(0)
    df['frame_diff'] = df['frame_diff'].astype(int)
    df['gaps'] = 0
    df.loc[df['frame_diff'] > 1,'gaps'] = df['frame_diff'] - 1
    df = df.drop(columns=['frame_diff'])
    return df

# def compute_hmean(df,rank,max_count):
#     df_t = df.copy(deep=True)
#     df_t = df_t.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).mean()
#     df_t['hmean'] = (3 * (df_t['iou'] * (df_t['count']/max_count)*df_t['most_consecutive_frames'])) / ((df_t['iou'] *(df_t['count']/max_count))+((df_t['count']/max_count)*df_t['most_consecutive_frames'])+(df_t['iou']*df_t['most_consecutive_frames']))
#     df_t['ranking'] = df_t['hmean'].rank(method='first',ascending=False)
#     df_t = df_t.sort_values(by='ranking')
#     df_t = df_t[df_t.ranking <= rank]
#     return df_t

def compute_hmean(df,rank,max_count):
    df_t = df.copy(deep=True)
    df_t = df_t.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).mean()
    df_t['hmean'] = (3 * (df_t['iou'] * (df_t['count']/max_count)*(df_t['most_consecutive_frames']/df_t['count']))) / ((df_t['iou'] *(df_t['count']/max_count))+((df_t['count']/max_count)*(df_t['most_consecutive_frames']/df_t['count']))+(df_t['iou']*(df_t['most_consecutive_frames']/df_t['count'])))
    df_t['ranking'] = df_t['hmean'].rank(method='first',ascending=False)
    df_t = df_t.sort_values(by='ranking')
    df_t = df_t[df_t.ranking <= rank]
    return df_t

def compute_hmean_4(df,rank,max_count,num_tracks):
    df_t = df.copy(deep=True)
    df_t = df_t.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).mean()

    iou = df_t['iou']
    count = df_t['count']/max_count
    most_consecutive_frames = df_t['most_consecutive_frames']/df_t['count']
    filtered_tracks = df_t['filtered_tracks']/num_tracks
    df_t['hmean'] = 4 / (1/iou + 1/count + 1/most_consecutive_frames + 1/filtered_tracks)
    df_t['ranking'] = df_t['hmean'].rank(method='first',ascending=False)
    df_t = df_t.sort_values(by='ranking')
    df_t = df_t[df_t.ranking <= rank]
    return df_t

# def get_most_consecutive_frames(df):
#     consecutive_fams = [list(group) for group in consecutive_groups(df.frame.values)]
#     l = [len(i) for i in consecutive_fams]
#     return max(l)

# def add_consecutive_frame(df_50_2_b,concentration,dt,sweep):
#     for i in df_50_2_b.iloc:
#         df = pd.read_csv(f'/tungstenfs/scratch/ggiorget/nessim/snakemake/benchmark_tracking/results/{concentration}pM/dt_{dt}/{sweep}/track_cost_cutoff_{i.track_cost_cutoff}_gap_closing_cost_cutoff_{i.gap_closing_cost_cutoff}_gap_closing_max_frame_count_{int(i.gap_closing_max_frame_count)}_seq_{int(i.seq)}.csv')
#         df = df[df.track_id == df_50_2_b.iloc[i.name].track_id1]
#         df_50_2_b.loc[i.name,'most_consecutive_frames'] = get_most_consecutive_frames(df)
#     return df_50_2_b

def add_consecutive_frame(df_50_2_b,concentration,dt,sweep):
    df_gt = pd.read_csv(f'/tungstenfs/scratch/ggiorget/nessim/snakemake/benchmark_tracking/results/{concentration}pM/dt_{dt}/{sweep}/gt_{concentration}pM_seq_0.csv')
    for i in df_50_2_b.iloc:
        # get the track id of the Gt and the exp
        gt_id = i.track_id2
        exp_id = i.track_id1
        # open the dataframes
        df_exp = pd.read_csv(f'/tungstenfs/scratch/ggiorget/nessim/snakemake/benchmark_tracking/results/{concentration}pM/dt_{dt}/{sweep}/track_cost_cutoff_{i.track_cost_cutoff}_gap_closing_cost_cutoff_{i.gap_closing_cost_cutoff}_gap_closing_max_frame_count_{int(i.gap_closing_max_frame_count)}_seq_{int(i.seq)}.csv')
        # get the number of gaps allowed
        n_gaps = int(i.gap_closing_max_frame_count)
        # rename the column x in y for the exp
        df_exp.rename(columns={'x':'a'},inplace=True)
        df_exp.rename(columns={'y':'x'},inplace=True)
        df_exp.rename(columns={'a':'y'},inplace=True)
        # merge the two dataframes on frame and x and y to get the spots in common
        df_m = pd.merge(df_gt[df_gt.track_id == gt_id],df_exp[df_exp.track_id == exp_id],on=['frame','x','y'])
        # get all the stretches of consecutive frames
        consecutive_fams = sorted(df_m.frame.values)
        # depending on the gaps allowed, get the longest stretch
        c_group = [[0]]
        for j in consecutive_fams:
            if j - c_group[-1][-1] < n_gaps:
                c_group[-1].append(j)
            else:
                c_group.append([j])

        longest_stretch = max([len(j) for j in c_group])

        df_50_2_b.loc[i.name,'most_consecutive_frames'] = longest_stretch
    return df_50_2_b

def plot_heatmap2D(df,col,metric,column):
    fig,ax = plt.subplots(2,col,figsize=(30,10))
    
    if metric == 'mean':
        df_heat = df.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).mean()[column].unstack(level=2).copy(deep=True)
    elif metric == 'max':
        df_heat = df.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).max()[column].unstack(level=2).copy(deep=True)
    elif metric == 'median':
        df_heat = df.groupby(['track_cost_cutoff','gap_closing_cost_cutoff','gap_closing_max_frame_count']).median()[column].unstack(level=2).copy(deep=True)
    
    counter = 0
    for l,i in enumerate(df_heat.columns):
        if column == 'count':
            sns.heatmap(df_heat[i].unstack(level=1),cmap='rocket_r',annot=True,vmin=0,ax=ax[l//col,counter])
        elif column == 'hmean':
            sns.heatmap(df_heat[i].unstack(level=1),cmap='rocket_r',annot=True,vmin=0,vmax=1,ax=ax[l//col,counter])
        else:
            sns.heatmap(df_heat[i].unstack(level=1),cmap='rocket_r',annot=True,vmin=0,vmax=1,ax=ax[l//col,counter])

        ax[l//col,counter].set_title(f'gap_closing_max_frame_count value : {i}')
        if counter == (col-1):
            counter = 0
        else:
            counter +=1

    fig.suptitle(f'Heatmap with computed {metric} {column} values',fontsize=20)
    fig.tight_layout()

def format_results(df,concentration:int,dt:int,sweep:str,max_count:int,threshold:int=10,rank:int=10000):

    df['iouw'] = df['iou'] * df['count']

    df_best = df[df.iouw > threshold]

    df_best.reset_index(drop=True,inplace=True)

    df_best = add_consecutive_frame(df_best,concentration=concentration,dt=dt,sweep=sweep)

    df_best_h = compute_hmean(df_best,rank=rank,max_count=max_count)

    return df_best,df_best_h

def check_filter(df_b,concentration,dt,sweep,t,g,m):
    path = f'/tungstenfs/scratch/ggiorget/nessim/snakemake/benchmark_tracking/results/{concentration}pM/dt_{dt}/{sweep}/'
    fil = f'track_cost_cutoff_{t}_gap_closing_cost_cutoff_{g}_gap_closing_max_frame_count_{m}'
    
    df_tracks_50_best = pd.DataFrame()
    counter = 0
    for file in os.listdir(path):
        if fil in file:
            df = pd.read_csv(os.path.join(path,file))
            df['seq'] = [counter]*len(df)
            df_tracks_50_best = pd.concat([df_tracks_50_best,df])
            counter +=1
    fig,ax = plt.subplots(figsize=(8,5))
    for i in range(len(df_b.seq.unique())):
        df_filter = df_b[(df_b.seq == i) & (df_b.track_cost_cutoff == t) & (df_b.gap_closing_cost_cutoff == g) & (df_b.gap_closing_max_frame_count == m)]
        if df_filter.empty:
            l = [1000]
        else:
            l = df_filter.track_id1.unique()
        f = df_tracks_50_best[df_tracks_50_best.seq == i].copy(deep=True).groupby('track_id').size()
        f = pd.DataFrame(f)
        f['in?'] = f.index.isin(l)
        f.reset_index(drop=True,inplace=True)
        sns.histplot(data=f,x=0,hue='in?')
        fig.suptitle(f'concentration {concentration} pM \n track_cost_cutoff:{t} \n gap_closing_cost_cutoff:{g} \n gap_closing_max_frame_count:{m}')
        ax.set_xlabel('track length')
        plt.legend(['True','False'],title = 'Filtered \n tracks')
        # plt.ylim(0,10)
        fig.tight_layout()

def check_filter_count(df_b,concentration,dt,sweep,t,g,m):
    path = f'/tungstenfs/scratch/ggiorget/nessim/snakemake/benchmark_tracking/results/{concentration}pM/dt_{dt}/{sweep}/'
    fil = f'track_cost_cutoff_{t}_gap_closing_cost_cutoff_{g}_gap_closing_max_frame_count_{m}'
    a = pd.DataFrame()
    df_tracks_50_best = pd.DataFrame()
    counter = 0
    for file in os.listdir(path):
        if fil in file:
            df = pd.read_csv(os.path.join(path,file))
            df['seq'] = [counter]*len(df)
            df_tracks_50_best = pd.concat([df_tracks_50_best,df])
            counter +=1
    for i in range(len(df_b.seq.unique())):
        df_filter = df_b[(df_b.seq == i) & (df_b.track_cost_cutoff == t) & (df_b.gap_closing_cost_cutoff == g) & (df_b.gap_closing_max_frame_count == m)]
        if df_filter.empty:
            l = [1000]
        else:
            l = df_filter.track_id1.unique()
        f = df_tracks_50_best[df_tracks_50_best.seq == i].copy(deep=True).groupby('track_id').size()
        f = pd.DataFrame(f)
        f['in?'] = f.index.isin(l)
        a = pd.concat([a,f])
        # find the maximum length of the tracks that are not in the filtered tracks
        # check if there is a track that should be filtered larger that the maximum length of the tracks that are not in the filtered tracks
    threshold = a[a['in?'] == False].max().values[0]
    g = a[a['in?'] == True] > a[a['in?'] == False].max()
    g = g[0]
    number_of_tracks_filtered = len(g[g == True])
    return number_of_tracks_filtered,threshold


def format_results_complete(df,concentration:int,dt:int,sweep:str,max_count:int,threshold:int=10,rank:int=10000):

    df['iouw'] = df['iou'] * df['count']

    df_best = df[df.iouw > threshold]

    df_best.reset_index(drop=True,inplace=True)

    df_best = add_consecutive_frame(df_best,concentration=concentration,dt=dt,sweep=sweep)

    df_best_h = compute_hmean(df_best,rank=rank,max_count=max_count)

    filt_tracks = [check_filter_count(df_best,concentration,dt,sweep,i.name[0],i.name[1],i.name[2])[0] for i in df_best_h.iloc]
    threshold = [check_filter_count(df_best,concentration,dt,sweep,i.name[0],i.name[1],i.name[2])[1] for i in df_best_h.iloc]

    df_best_h['filtered_tracks'] = filt_tracks
    df_best_h['threshold'] = threshold

    df_filt = df_best_h[df_best_h.filtered_tracks > 0]

    num_tracks = len(df_best.track_id2.unique())

    df_best_h = compute_hmean_4(df_filt,rank=rank,max_count=max_count,num_tracks=num_tracks)
    
    return df_best,df_best_h,df_filt