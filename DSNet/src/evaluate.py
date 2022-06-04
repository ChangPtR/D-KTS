import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model
from time import time
import h5py
import os.path as osp

logger = logging.getLogger()

# python evaluate.py anchor-free --model-dir ../models/tvsum30sift/ --splits ../dataset_split/tvsum_splits.yml --freq 1 --base-model linear --num-feature 128 --dataset ../my_data/tvsum30_sift.h5


def evaluate(model, val_loader, args):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')
    # d3 = h5py.File('../../my_data/summe30_googlenet.h5', 'r')
    d3 = h5py.File('../../my_data/tvsum30_googlenet.h5', 'r')
    times={}
    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks,_ in val_loader:
            user_summary = d3[test_key+'/user_summary'][()]
            st = time()

            if seq.shape[0]>picks.shape[0]:
                seq=seq[:-1]

            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)

            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            ed=time()
            # print(test_key, ed - st)

            times[test_key]=ed-st
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)

            pred_summ = vsumm_helper.downsample_summ(pred_summ,args.freq)

            # print(pred_summ.shape,seq_len)
            if pred_summ.shape[0] > seq_len:
                pred_summ = pred_summ[:seq_len]

            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)
    d3.close()
    return stats.fscore, stats.diversity,times


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(args)
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(args.dataset,split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            fscore, diversity,times = evaluate(model, val_loader, args)
            stats.update(fscore=fscore, diversity=diversity)

            np.save(args.model_dir+'/time%d.npy'%split_idx, times)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()


