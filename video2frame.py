import cv2
import os
import h5py
from time import time
import argparse

# nohup python -u video2frame.py >> log_summe/v2f.o &

def one_output():

    s='DSN/models/tvsum/result4.h5'
    # s='DSN/models/summe/result4.h5'
    f = h5py.File(s, 'a')
    print(s)
    for key in f.keys():
        if key != 'mean_fm':
            mv='../cby_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/'+key
            folder='results/tvsum/'+key[:11]+'/frames/'
            # mv='../cby_data/SumMe/videos/'+key
            # folder='results/summe/'+os.path.splitext(key)[0]+'/frames/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            machine_summary = f[key+'/machine_summary'][()]
            st = time()
            camera = cv2.VideoCapture(mv)
            times = 0
            while True:
                res, image = camera.read()
                if not res:
                    print(times)
                    break
                if machine_summary[times] == 1:
                    cv2.imwrite(folder + str(times).zfill(6) + '.jpg', image)
                times += 1
            ed = time()
            f.create_dataset(key + '/time4.1', data=ed - st)
            print(ed - st)

one_output()
