import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
from time import time

# nohup python -u summary2video.py >> log_summe/vs2v.o &

parser = argparse.ArgumentParser()

parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")
# parser.add_argument('--save-dir', type=str, help="directory to save")
# parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

def frm2video(frm_dir,vid_writer):
    files = os.listdir(frm_dir)
    for file in files:
        frm = cv2.imread(osp.join(frm_dir,file))
        frm = cv2.resize(frm, (args.width, args.height))
        vid_writer.write(frm)

def one():

    s = 'DSN/models/tvsum/result4.h5'
    # s = 'DSN/models/summe/result4.h5'
    f = h5py.File(s, 'a')
    print(s)
    for key in f.keys():
        if key != 'mean_fm':
            folder = 'results/tvsum/'+ key[:11] + '/frames/'
            save_dir = 'results/tvsum/' + key[:11] + '/'
            vname=os.path.splitext(key)[0]
            # folder = 'results/summe/'+ vname + '/frames/'
            # save_dir ='results/summe/' + vname+'/'
            print(key)
            st = time()
            vid_writer = cv2.VideoWriter(
                osp.join(save_dir, 'vs_'+key),
                cv2.VideoWriter_fourcc(*'mp4v'),
                args.fps,
                (args.width, args.height),
            )
            frm2video(folder,vid_writer)
            ed = time()
            print(ed - st)

            f.create_dataset(key + '/time4.2', data=ed - st)
            vid_writer.release()
    f.close()

one()

