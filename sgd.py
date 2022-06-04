import os
import h5py
from DSN.utils import read_json
import numpy as np
import torch
import matplotlib.pyplot as plt

EXTRACT_FOLDER = '../cby_data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video'

def find():
    splits=read_json('dataset_split/tvsum_splits.json')
    tests=[]
    for i in range(5):
        tests.extend(splits[i]['test_keys'])
    print(len(tests))
    files = os.listdir(EXTRACT_FOLDER)
    for file in files:
        if file not in tests:
            print(file)

def getM():
    fg=h5py.File('my_data/tvsum5_googlenet.h5', 'r')
    fo=h5py.File('my_data/orb51.h5', 'r')

    net = torch.nn.Linear(32, 1024)
    torch.nn.init.normal_(net.weight,  mean=0, std=0.01)
    torch.nn.init.constant_(net.bias, val=0)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()

    arr=[]
    for epoch in range(10):
        for key in ['-esJrBWj2d8.mp4','z_6gVvQb2d0.mp4']: # 5% of TVSum dataset
            gg=fg[key+'/features'][()]
            cv=fo[key+'/features'][()]
            for i in range(gg.shape[0]-1):
                x1=torch.from_numpy(cv[i]).float()
                x2=torch.from_numpy(cv[i+1]).float()
                g1=torch.from_numpy(gg[i]).float()
                g2=torch.from_numpy(gg[i+1]).float()

                pred = net(x2 - x1)
                l = loss(pred,g2-g1)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
        print(epoch,'loss: ',l.item())
        arr.append(l.item())

    torch.save(net.state_dict(), 'mix_feature/M.pth')
    plt.plot(arr)
    plt.savefig('loss.jpg')

getM()
# find()
