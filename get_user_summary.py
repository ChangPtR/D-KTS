import h5py
import numpy as np
from DSN.knapsack import knapsack_dp
import math
import argparse
import os
import scipy.io as scio

def make_user_summary(frame_scores, cps, n_frames, nfps, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - frame_scores: importance scores by users.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    n_segs = cps.shape[0]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        # print(start,end)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros(n_frames, dtype=np.float32)
    for seg_idx in picks:
        first, last = cps[seg_idx]
        summary[first:last + 1] = 1

    return summary

# nohup python -u get_user_summary.py --frequency 15 --file summe2_googlenet.h5 >> log/us.o &

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
parser.add_argument('--frequency', type=int, help="downsample frequency")
parser.add_argument('--file', type=str, help="h5py file name to save features")
args = parser.parse_args()

H5_FILE='my_data/'+args.file
EXTRACT_FREQUENCY = args.frequency
d=h5py.File(H5_FILE, 'a')
print('========== fps:',30/EXTRACT_FREQUENCY,'==========')

dataFile = '../cby_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat'
data_folder = '../cby_data/SumMe/GT/'

def get_summe():

    mats=os.listdir(data_folder)
    for mt in mats:
        print(os.path.join(data_folder,mt))
        mat = scio.loadmat(os.path.join(data_folder,mt))
        uscore_idx=mat['user_score'].T

        gtscore_idx=mat['gt_score']
        gt_score=np.squeeze(gtscore_idx)
        gt_score=gt_score[::EXTRACT_FREQUENCY]

        mt=os.path.splitext(mt)[0]+".mp4"
        print(mt, uscore_idx.shape,gt_score.shape,d[mt+'/picks'][()].shape)

        if d[mt+'/picks'][()].shape[0]<gt_score.shape[0]:
            gt_score=gt_score[:d[mt+'/picks'][()].shape[0]]
        if d[mt+'/picks'][()].shape[0]>gt_score.shape[0]:
            np.pad(gt_score,(0,d[mt+'/picks'][()].shape[0]-gt_score.shape[0]),'constant')

        user_summary = []
        cps = d[mt+'/change_points'][()]
        nfps = d[mt+'/n_frame_per_seg'][()].tolist()
        n_frames = d[mt+'/n_frames'][()]
        for us in uscore_idx:
            one_sum=make_user_summary(us, cps, n_frames, nfps)
            user_summary.append(one_sum)

        d.create_dataset(mt + '/gt_score', data=gt_score)
        d.create_dataset(mt+'/user_summary', data=user_summary)

def get_tvsum():
    mat = h5py.File(dataFile, 'r')
    for i in range(50):
        uscore_idx = mat['tvsum50/user_anno'][i, 0]
        user_scores = mat[uscore_idx]

        gtscore_idx = mat['tvsum50/gt_score'][i, 0]
        gt_score = np.squeeze(mat[gtscore_idx])
        gt_score = gt_score[::EXTRACT_FREQUENCY]

        name_idx = mat['tvsum50/video'][i, 0]
        video_name = "".join(chr(i) for i in mat[name_idx][()]) + ".mp4"

        if d[video_name+'/picks'][()].shape[0]<gt_score.shape[0]:
            gt_score=gt_score[:d[video_name+'/picks'][()].shape[0]]
        if d[video_name+'/picks'][()].shape[0]>gt_score.shape[0]:
            np.pad(gt_score,(0,d[video_name+'/picks'][()].shape[0]-gt_score.shape[0]),'constant')

        print(i, user_scores.shape,gt_score.shape,d[video_name+'/picks'][()].shape)

        user_summary = []
        cps = d[video_name + '/change_points'][()]
        nfps = d[video_name + '/n_frame_per_seg'][()].tolist()
        n_frames = d[video_name + '/n_frames'][()]
        for us in user_scores:
            one_sum = make_user_summary(us, cps, n_frames, nfps)
            user_summary.append(one_sum)

        d.create_dataset(video_name + '/gt_score', data=gt_score)
        d.create_dataset(video_name + '/user_summary', data=user_summary)

# get_summe()
get_tvsum()

d.close()