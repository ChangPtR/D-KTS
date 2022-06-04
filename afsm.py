import numpy as np
import cv2
import os
import h5py
import torch
import torchvision.models as models
from torch.autograd import Variable
from torchvision.transforms import transforms
from time import time


# nohup python -u afsm.py >> mix_feature/ex_time.o &

EXTRACT_FREQUENCY = 6
EXTRACT_FOLDER = '../cby_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video'

googlenet = models.googlenet(pretrained=True)
googlenet=torch.nn.Sequential(*list(googlenet.children())[:-2])
googlenet.eval()

M=torch.nn.Linear(32, 1024)
M.load_state_dict(torch.load('mix_feature/M.pth'))
M.eval()

def orb(path):
    cap = cv2.VideoCapture(path)
    count=0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_features=[]
    orb = cv2.ORB_create()
    base_google=np.zeros((1024))
    base_cv=torch.zeros((32))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, fr = cap.read()
        if ret is False:break
        count += 1
        if count % EXTRACT_FREQUENCY == 0:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            kp = orb.detect(gray, None)
            kp, des = orb.compute(gray, kp)
            flag=True
            if des is not None and count!=EXTRACT_FREQUENCY:
                if des.shape[0]<=1000:
                    flag=False
                    des1 = torch.from_numpy(np.mean(des, 0))
                    # variable = Variable(torch.from_numpy(des1).float(), requires_grad=False)
                    r=torch.squeeze(M(base_cv.float()-des1.float())).detach().numpy()
                    feature=base_google+r

            if count==EXTRACT_FREQUENCY or flag:
                tensor = transforms.ToTensor()(fr)
                variable = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
                feature = torch.squeeze(googlenet(variable)).detach().numpy()
                base_google=feature
                if des is not None:
                    base_cv=torch.from_numpy(np.mean(des, 0))

            video_features.append(feature)
    cap.release()
    return np.array(video_features),frame_count,fps


f = h5py.File('my_data/tvsum5_mix.h5', 'w')
files = os.listdir(EXTRACT_FOLDER)
cnt=0
ts=0
for file in files:
    path = EXTRACT_FOLDER + "/" + file
    st=time()
    video_features, fcnt, fps = orb(path)
    duration = fcnt / fps
    ed = time()
    cnt += 1
    ts=ts+ed-st
    print(cnt, file, ed - st, video_features.shape,fcnt)
    f.create_dataset(file + '/n_frames', data=int(fcnt))
    f.create_dataset(file + '/features', data=video_features)
    picks = np.arange(0, video_features.shape[0]) * EXTRACT_FREQUENCY
    f.create_dataset(file + '/picks', data=picks)
    f.create_dataset(file + '/time1', data=ed - st)
    f.create_dataset(file + '/duration', data=duration)

f.close()

print('total time: ',ts)


