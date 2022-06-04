import torch
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import cv2
import os
import h5py
from time import time
import argparse
from skimage.feature import local_binary_pattern
# import imagehash
# from PIL import Image
from torchvision.transforms import transforms

# python feature_extractor.py --frequency 15 --file tvsum2_googlenet.h5
# nohup python -u feature_extractor.py --frequency 15 --file summe2_googlenet.h5 >> log/extract2.o &

parser = argparse.ArgumentParser(
    "Pytorch code for unsupervised video summarization with REINFORCE")
parser.add_argument('--frequency', type=int, help="downsample frequency",default=1)
parser.add_argument('--file', type=str, help="h5py file name to save features")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--t', type=float, default=4.0)
# parser.add_argument('--c', type=int, default=5)
args = parser.parse_args()

# EXTRACT_FOLDER = 'E:/vs_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video'
EXTRACT_FOLDER = '../cby_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video'
# EXTRACT_FOLDER = '../cby_data/SumMe/videos'

EXTRACT_FREQUENCY = args.frequency
BATCH_SIZE = 16

googlenet = models.googlenet(pretrained=True)
googlenet = torch.nn.Sequential(*list(googlenet.children())[:-2])
# print(googlenet)
googlenet.eval()
if not args.use_cpu:
    googlenet = googlenet.cuda()
else:
    print('Using CPU......')

def dhash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, len(dhash_str), 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print(result)
    return result

def hanming(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def one_video_hash(path, threshold):
    cap = cv2.VideoCapture(path)
    frames=[]
    video_features = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    skip_count = 0
    arr = []

    with torch.no_grad():
        base = None
        hash1 = None
        while cap.isOpened():
            # Capture frame-by-frame
            ret, fr = cap.read()
            if ret is False:
                break
            count += 1

            if count % EXTRACT_FREQUENCY == 0:
                hash2=dhash(fr)
                if hash1 is not None:
                    dist = hanming(hash1,hash2)
                if base is None or dist > threshold:
                    base = fr
                    hash1 = hash2
                    frames.append(np.rollaxis(fr, 2))
                    arr.append(skip_count)
                    skip_count = 0
                else:
                    skip_count += 1
                    frames.append(np.rollaxis(base, 2))
                if (len(frames) == BATCH_SIZE) or \
                    (count == frame_count//EXTRACT_FREQUENCY * EXTRACT_FREQUENCY and len(frames) > 0):
                    batch = np.array(frames)
                    if args.use_cpu:
                        variable = Variable(torch.from_numpy(batch).float())
                        feature = googlenet(variable).detach().numpy()
                    else:
                        variable = Variable(torch.from_numpy(batch).float()).cuda()
                        feature = googlenet(variable).cpu().detach().numpy()
                    video_features.extend(feature)
                    frames.clear()

    cap.release()
    video_features = np.squeeze(np.array(video_features))
    return video_features, frame_count, fps,arr

# def one_video_my(path, threshold, c):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     kts_frames=[]
#     video_features = []
#     kts_feature=[]
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     count = 0
#     skip_count = 0
#     arr = []
#     index=[]
#     with torch.no_grad():
#         base = None
#         hash1 = None
#         while cap.isOpened():
#             # Capture frame-by-frame
#             ret, fr = cap.read()
#             if ret is False:
#                 break
#             count += 1
#
#             if count % EXTRACT_FREQUENCY == 0:
#                 hash2 = dhash(fr)
#                 if hash1 is not None:
#                     dist = hanming(hash1, hash2)
#                 if base is None or dist > threshold or skip_count >= c:
#                     base = fr
#                     hash1 = hash2
#                     frames.append(np.rollaxis(fr, 2))
#                     kts_frames.append(np.rollaxis(fr, 2))
#                     arr.append(skip_count)
#                     skip_count = 0
#                     index.append(count-1)
#                 else:
#                     skip_count += 1
#                     frames.append(np.rollaxis(base, 2))
#                 if (len(frames) == BATCH_SIZE) or \
#                         (count == frame_count // EXTRACT_FREQUENCY * EXTRACT_FREQUENCY and len(frames) > 0):
#                     batch = np.array(frames)
#                     if args.use_cpu:
#                         variable = Variable(torch.from_numpy(batch).float())
#                         feature = googlenet(variable).detach().numpy()
#                     else:
#                         variable = Variable(torch.from_numpy(batch).float()).cuda()
#                         feature = googlenet(variable).cpu().detach().numpy()
#                     video_features.extend(feature)
#                     # kts_feature.extend(feature)
#                     frames.clear()
#
#     cap.release()
#     video_features = np.squeeze(np.array(video_features))
#     # kts_feature = np.squeeze(np.array(kts_feature))
#     return video_features, frame_count, fps, arr

def one_video(path):
    cap = cv2.VideoCapture(path)
    count = 0
    frames = []
    video_features = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with torch.no_grad():
        while cap.isOpened():
            # Capture frame-by-frame
            ret, fr = cap.read()
            if ret is False:
                break
            count += 1
            if count % EXTRACT_FREQUENCY == 0:
                frames.append(np.rollaxis(fr, 2))
            if (len(frames) == BATCH_SIZE) or \
                    (count == frame_count//EXTRACT_FREQUENCY * EXTRACT_FREQUENCY and len(frames) > 0):
                batch = np.array(frames)
                if args.use_cpu:
                    variable = Variable(torch.from_numpy(batch).float())
                    feature = googlenet(variable).detach().numpy()
                else:
                    variable = Variable(torch.from_numpy(batch).float()).cuda()
                    feature = googlenet(variable).cpu().detach().numpy()
                video_features.extend(feature)
                frames.clear()
    cap.release()

    video_features = np.squeeze(np.array(video_features))

    return video_features, frame_count, fps


# def sift(path):
#     cap = cv2.VideoCapture(path)
#     count = 0
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     video_features = []
#     # sift = cv2.SIFT_create()
#     sift = cv2.xfeatures2d.SIFT_create()
#     while cap.isOpened():
#         # Capture frame-by-frame
#         ret, fr = cap.read()
#         if ret is False:
#             break
#         count += 1
#         if count % EXTRACT_FREQUENCY == 0:
#             gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#             kp, des = sift.detectAndCompute(gray, None)
#             des1 = np.zeros((128))
#             if des is not None:
#                 des1 = np.mean(des, 0)
#             video_features.append(des1)
#     cap.release()
#
#     return np.array(video_features), frame_count, fps
# def orb(path):
#     cap = cv2.VideoCapture(path)
#     count = 0
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     video_features = []
#     orb = cv2.ORB_create(nfeatures=2000)
#     arr = []
#     while cap.isOpened():
#         # Capture frame-by-frame
#         ret, fr = cap.read()
#         if ret is False:
#             break
#         count += 1
#         if count % EXTRACT_FREQUENCY == 0:
#             gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#             # kp = orb.detect(gray, None)
#             kp = orb.detect(gray, None)
#             kp, des = orb.compute(gray, kp)
#             des1 = np.zeros((32))
#             if des is not None:
#                 arr.append(des.shape[0])
#                 des1 = np.mean(des, 0)
#             else:
#                 arr.append(0)
#
#             video_features.append(des1)
#     cap.release()
#
#     return np.array(video_features), frame_count, fps, arr
# def lbp(path):
#     cap = cv2.VideoCapture(path)
#     count = 0
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     video_features = []
#     while cap.isOpened():
#         # Capture frame-by-frame
#         ret, fr = cap.read()
#         if ret is False:
#             break
#         count += 1
#         if count % EXTRACT_FREQUENCY == 0:
#             gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#             img_ku = local_binary_pattern(gray, 8, 1, method='default')
#             max_bins = int(img_ku.max() + 1)
#             hist, _ = np.histogram(img_ku, density=True,
#                                    bins=max_bins, range=(0, max_bins))
#             # print(hist.shape)
#             video_features.append(hist)
#     cap.release()
#
#     return np.array(video_features), frame_count, fps

h5_file = 'my_data/'+args.file
f = h5py.File(h5_file, 'w')
files = os.listdir(EXTRACT_FOLDER)
cnt = 0

for file in files:
    cnt += 1
    path = EXTRACT_FOLDER + "/" + file
    st = time()
    # video_features, fcnt, fps = one_video(path)
    video_features, fcnt, fps,skp_arr = one_video_hash(path,args.t)
    # video_features, fcnt, fps, skp_arr, kts_feature,index = one_video_my(
    #     path, args.threshold, args.c)
    # video_features,fcnt,fps = sift(path)
    # video_features, fcnt, fps,arr= orb(path)
    # video_features, fcnt, fps = lbp(path)
    duration = fcnt/fps
    ed = time()

    print(cnt, file, ed-st, fcnt,video_features.shape)

    f.create_dataset(file + '/n_frames', data=int(fcnt))
    f.create_dataset(file + '/features', data=video_features)
    picks = np.arange(0, video_features.shape[0]) * EXTRACT_FREQUENCY
    f.create_dataset(file + '/picks', data=picks)
    f.create_dataset(file+'/time1', data=ed-st)
    f.create_dataset(file+'/duration', data=duration)

    f.create_dataset(file + '/skip_arr', data=skp_arr)

    # f.create_dataset(file + '/kts_feature', data=kts_feature)
    # f.create_dataset(file + '/index', data=index)

f.close()
