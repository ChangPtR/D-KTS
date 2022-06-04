import logging
import torch
import h5py
import sys
import numpy as np

logger = logging.getLogger()

from anchor_free import anchor_free_helper
from anchor_free.dsnet_af import DSNetAF
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
# import anchor_free_helper

sys.path.append('../')
from helpers import data_helper, vsumm_helper
from evaluate import evaluate

from torch.utils.data import DataLoader



# def collate_fn(batch):
#
#     keys = [item[0] for item in batch]
#     seqs=[item[1] for item in batch]
#     gtscores=[item[2] for item in batch]
#     cpss = [item[3] for item in batch]
#     n_framess = [item[4] for item in batch]
#     nfpss = [item[5] for item in batch]
#     pickss = [item[6] for item in batch]
#     user_summarys = [item[7] for item in batch]
#
#     return np.asarray(keys), np.asarray(seqs), np.asarray(gtscores), \
#            np.asarray(cpss), np.asarray(n_framess), np.asarray(nfpss), np.asarray(pickss), np.asarray(user_summarys)

def train(args, split, save_path):
    # device_ids = [2,3]

    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                        num_hidden=args.num_hidden, num_head=args.num_head)

    # model=torch.nn.DataParallel(model, device_ids)
    # device = torch.device(args.device)
    # model = model.to(device)
    if args.device=='cuda':
        model=model.cuda()

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # optimizer = torch.nn.DataParallel(optimizer, device_ids)

    max_val_fscore = -1

    train_set = data_helper.VideoDataset(args.dataset,split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=4,shuffle=True,collate_fn=collate_fn)

    val_set = data_helper.VideoDataset(args.dataset,split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'loc_loss',
                                         'ctr_loss')

        for key, seq, gtscore, change_points, n_frames, nfps, picks, _ in train_loader:
            if picks.shape>gtscore.shape:
                picks=picks[:gtscore.shape[0]]

            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ,args.freq)
            if not target.any():
                continue
            if target.shape[0] > seq.shape[0]:
                target = target[:seq.shape[0]]

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            cls_label = target
            loc_label = anchor_free_helper.get_loc_label(target)
            ctr_label = anchor_free_helper.get_ctr_label(target, loc_label)

            pred_cls, pred_loc, pred_ctr = model(seq)

            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)
            ctr_label = torch.tensor(ctr_label, dtype=torch.float32).to(args.device)

            cls_loss = calc_cls_loss(pred_cls, cls_label, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label,
                                     args.reg_loss)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)

            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item())
            
            #[Ke] Clean the gpu cache
            del seq, keyshot_summ, target, pred_cls, pred_ctr, pred_loc
            torch.cuda.empty_cache()

        val_fscore, _, times = evaluate(model, val_loader, args)
        np.save(args.model_dir + '/time.npy', times)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.loss:.4f} '
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    return max_val_fscore

