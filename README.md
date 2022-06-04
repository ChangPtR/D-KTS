# D-KTS
Code for paper [TOWARDS PRACTICAL AND EFFICIENT LONG VIDEO SUMMARY](https://ieeexplore.ieee.org/abstract/document/9746911). Supervised and unsupervised video summarization systems based on our Distribution-based KTS.

#### Architecture

1. Feature extraction
2. Kernel Temporal Segmentation (KTS)
3. Model inference
4. Summary video generation
   1. Extract the predicted frames from the video.
   2. Compose summary video with frames.

#### Get Started

This project is developed on Ubuntu 16.04. 
The main requirements are Python 3.6, Anaconda 4.8, PyTorch 1.4.0, cudatoolkit 10.0, opencv 3.4.2, torchvision 0.5.0.

First, clone this project to your local environment.

```
git clone https://gitee.com/rainemo/vs-pipline.git
```

Next, download [SumMe](https://gyglim.github.io/me/vsum/index.html#benchmark) and [TVSum](https://github.com/yalesong/tvsum) datasets into a folder.

#### Feature Extraction

First, change the `EXTRACT_FOLDER` in feature_extractor.py to your own folder. Then run

```buildoutcfg
# down sample to 2 fps
python feature_extractor.py --frequency 15 --file tvsum2_googlenet.h5
```

The features will be saved to h5py file in `my_data/` folder.
Next, randomly split datasets for 5-fold cross validation.

```buildoutcfg
# for DSN, return json
python create_split.py -d ../my_data/tvsum2_googlenet.h5 --save-dir dataset_split --save-name tvsum_splits --num-splits 5
# for DSNet, return yaml
python make_split.py --dataset ../../my_data/tvsum2_googlenet.h5 --save-path ../dataset_split/tvsum_splits --num-splits 5
```

#### KTS

```buildoutcfg
python kts_run.py --frequency 15 --file tvsum2_googlenet.h5
```

You can also use `--v` to set the weight for penalty term, v = 1 by default.

#### Model Inference 

Models should be retrained for each change of the feature. We add evaluating method after each training method.
For the supervised model DSNet, you should run get_user_summary.py to add label for each video before training.

```buildoutcfg
# train and evaluate DSN
python main.py -d ../my_data/tvsum2_googlenet.h5 -s dataset_split/tvsum_splits.json -m tvsum --gpu 0 --save-dir tvsum2 --verbose

# add label(gt_score and user_summary) to the dataset
python get_user_summary.py --frequency 15 --file tvsum2_googlenet.h5
# train and evaluate DSNet
python train.py --dataset ../../my_data/tvsum2_googlenet.h5 --model-dir ../models/tvsum2 --splits ../dataset_split/tvsum_splits.yml --freq 15
```

To evaluate, only run

```buildoutcfg
#evaluate DSN
python main.py -d ../my_data/tvsum2_googlenet.h5 -s dataset_split/tvsum_splits.json -m tvsum --gpu 0 --save-dir tvsum2 --evaluate --resume path_to_your_model.pth.tar --verbose
# evaluate DSNet
python evaluate.py anchor-free --model-dir ../models/tvsum2/ --splits ../dataset_split/tvsum_splits.yml --freq 15 --base-model linear --dataset ../my_data/tvsum2_googlenet.h5
```

#### Summary Video Generation

First, select key frames according to the binary `machine_summary` and save them into a folder.
Second, generate summary video with key frames.
Before you run the code below, you should change the paths in video2frame.py and summary2video.py to your own.

```buildoutcfg
python video2frame.py
python summary2video.py
```

#### CV Features and AFSM

For LBP, we set v = 1e-05 in kys_run.py.
For ORB, we set v = 4 in kys_run.py, and DSN training parameters: 
learning rate = 1e-06, weight decay = 0.1, epoch = 20.

In AFSM, you can train the residual projection matrix with sgd.py. 
First, change the path to googlenet feature `fg` and ORB feature `fo` to your own.
Next, set the maximum epoch and learning rate. Then run

```buildoutcfg
python sgd.py
```

The matrix will be saved under `mix_feature/` folder. 
Before you run afsm.py as below, you can modify the `EXTRACT__FREQUENCY` and `EXTRACT_FOLDER`.

```buildoutcfg
python afsm.py
```

The mixed features will be saved to h5py file in `my_data/` folder.


#### Acknowledgments

- Thank [SumMe](https://gyglim.github.io/me/vsum/index.html#benchmark) and [TVSum](https://github.com/yalesong/tvsum) for the dataset.
- Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) and [DSNet](https://github.com/li-plus/DSNet) for the models.

