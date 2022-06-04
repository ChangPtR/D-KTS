import numpy as np
from kts.auto_alter import cpd_auto2
from kts.nonlin_alter import kernel
import h5py
from time import time
import math
import argparse

# nohup python -u kts_run.py --frequency 15 --file summe2_googlenet.h5 >> log/kts2.o &
# nohup python -u kts_run.py --frequency 6 --file orb51.h5 --v 2 >> mix_feature/kts.o &

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
parser.add_argument('--frequency', type=int, help="downsample frequency")
parser.add_argument('--file', type=str, help="h5py file name to save features")
parser.add_argument('--v', type=float, default=1.0,help="vmax in penalty")
args = parser.parse_args()

EXTRACT_FREQUENCY = args.frequency
H5_FILE= 'my_data/'+args.file

d=h5py.File(H5_FILE, 'a')

for key in d.keys():

    X = d[key+'/features'][()]
    n_frames = d[key + '/n_frames'][()]
    n = X.shape[0]
    n1 = min(n, 338) # 95%
    m = round(n_frames / 106 * 2)

    st1 = time()
    K1 = kernel(X, X.T, n1)
    cps1, scores1 = cpd_auto2(K1, m, args.v,EXTRACT_FREQUENCY)
    ed1 = time()
    cps1 *= EXTRACT_FREQUENCY
    cps1 = np.hstack((0, cps1, n_frames))
    begin_frames = cps1[:-1]
    end_frames = cps1[1:]
    cps1 = np.vstack((begin_frames, end_frames - 1)).T
    print(key, ed1 - st1, n, m, cps1.shape)

    d.create_dataset(key + '/time2', data=ed1 - st1)
    d.create_dataset(key+'/change_points', data=cps1)
    n_frame_per_seg = end_frames - begin_frames
    d.create_dataset(key+'/n_frame_per_seg',data=n_frame_per_seg)


d.close()

