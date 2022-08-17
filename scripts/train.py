
import jittor as jt
from jittor import init
'Script for multi-gpu training.'
import json
import os

# os.environ['JT_SYNC'] = '1'
# os.environ['trace_py_var'] = '3'
import sys
import numpy as np
from jittor import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord
# tycoer
assert opt.nThreads == 0, 'AlphaPose_jittor only support nThreads is 0.'
assert opt.sync == False, 'As nn.SyncBatchNorm is not implemented in Jittor, sync is not supportted for AlphaPose_jittor'
if jt.has_cuda:
    jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 0

num_gpu = jt.get_device_count()
valid_batch = (1 * num_gpu)
norm_layer = nn.BatchNorm2d

def train(opt, train_loader, m, criterion, optimizer, writer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    # train_loader = tqdm(train_loader, dynamic_ncols=True)
    bar = tqdm(range(len(train_loader) // train_loader.batch_size)) # tycoer
    for (i, (inps, labels, label_masks, _, bboxes)) in enumerate(train_loader):
        # if isinstance(inps, list):
        #     inps = [inp.requires_grad_() for inp in inps]
        # else:
        #     inps = inps.requires_grad_()
        if isinstance(inps, list):
            for inp in inps:
                inp.requires_grad = True
        else:
            inps.requires_grad = True

        if isinstance(labels, list):
            labels = [label for label in labels]
            label_masks = [label_mask for label_mask in label_masks]
        else:
            labels = labels
            label_masks = label_masks
        output = m(inps)
        if (cfg.LOSS.get('TYPE') == 'MSELoss'):
            loss = (0.5 * criterion(output * (label_masks), labels * (label_masks)))
            acc = calc_accuracy(output * (label_masks), labels * (label_masks))
        elif (cfg.LOSS.get('TYPE') == 'Combined'):
            if (output.shape[1] == 68):
                face_hand_num = 42
            else:
                face_hand_num = 110
            output_body_foot = output[:, :(- face_hand_num), :, :]
            output_face_hand = output[:, (- face_hand_num):, :, :]
            num_body_foot = output_body_foot.shape[1]
            num_face_hand = output_face_hand.shape[1]
            label_masks_body_foot = label_masks[0]
            label_masks_face_hand = label_masks[1]
            labels_body_foot = labels[0]
            labels_face_hand = labels[1]
            loss_body_foot = (0.5 * criterion[0](output_body_foot * (label_masks_body_foot), labels_body_foot * (label_masks_body_foot)))
            acc_body_foot = calc_accuracy(output_body_foot * (label_masks_body_foot), labels_body_foot * (label_masks_body_foot))
            loss_face_hand = criterion[1](output_face_hand, labels_face_hand, label_masks_face_hand)
            acc_face_hand = calc_integral_accuracy(output_face_hand, labels_face_hand, label_masks_face_hand, output_3d=False, norm_type=norm_type)
            loss_body_foot *= 100
            loss_face_hand *= 0.01
            loss = (loss_body_foot + loss_face_hand)
            acc = (((acc_body_foot * num_body_foot) / (num_body_foot + num_face_hand)) + ((acc_face_hand * num_face_hand) / (num_body_foot + num_face_hand)))
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)
        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)
        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.step(loss) # tycoer
        opt.trainIters += 1
        if opt.board:
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, 'Train')
        if (opt.debug and (not (i % 10))):
            debug_writing(writer, output, labels, inps, opt.trainIters)
        train_loader.set_description('loss: {loss:.8f} | acc: {acc:.4f}'.format(loss=loss_logger.avg, acc=acc_logger.avg))
        bar.update() # tycoer
    train_loader.close()
    return (loss_logger.avg, acc_logger.avg)

def validate(m, opt, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    # det_loader = jt.utils.data.DataLoader(det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    det_loader = det_dataset.set_attrs(batch_size=batch_size, shuffle=False, num_workers=opt.nThreads, drop_last=False) # tycoer

    kpt_json = []
    eval_joints = det_dataset.EVAL_JOINTS
    m.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    halpe = ((cfg.DATA_PRESET.NUM_JOINTS == 133) or (cfg.DATA_PRESET.NUM_JOINTS == 136))
    bar = tqdm(range(len(det_loader) // det_loader.batch_size))
    for (inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds) in det_loader:
        if isinstance(inps, list):
            inps = [inp for inp in inps]
        else:
            inps = inps
        output = m(inps)
        pred = output
        assert (pred.ndim == 4)
        pred = pred[:, eval_joints, :, :]
        if (output.shape[1] == 68):
            face_hand_num = 42
        else:
            face_hand_num = 110
        for i in range(output.shape[0]):
            bbox = crop_bboxes[i].tolist()
            if combined_loss:
                (pose_coords_body_foot, pose_scores_body_foot) = heatmap_to_coord[0](pred[i][det_dataset.EVAL_JOINTS[:(- face_hand_num)]], bbox, hm_shape=hm_size, norm_type=norm_type)
                (pose_coords_face_hand, pose_scores_face_hand) = heatmap_to_coord[1](pred[i][det_dataset.EVAL_JOINTS[(- face_hand_num):]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_scores = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                (pose_coords, pose_scores) = heatmap_to_coord(pred[i][det_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)
            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape((- 1)).tolist()
            data = dict()
            data['bbox'] = bboxes[(i, 0)].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(((scores[i] + np.mean(pose_scores)) + (1.25 * np.max(pose_scores))))
            data['category_id'] = 1
            data['keypoints'] = keypoints
            kpt_json.append(data)
        bar.update()
    sysout = sys.stdout
    with open(os.path.join(opt.work_dir, 'test_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt.json'), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN), halpe=halpe)
    sys.stdout = sysout
    return res

def validate_gt(m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS
    # gt_val_loader = jt.utils.data.DataLoader(gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    gt_val_loader = gt_val_dataset.set_attrs(batch_size=batch_size, shuffle=False, num_workers=opt.nThreads, drop_last=False) # tycoer

    kpt_json = []
    m.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    halpe = ((cfg.DATA_PRESET.NUM_JOINTS == 133) or (cfg.DATA_PRESET.NUM_JOINTS == 136))
    bar = tqdm(range(len(gt_val_loader) // gt_val_loader.batch_size))
    for (inps, labels, label_masks, img_ids, bboxes) in gt_val_loader:
        if isinstance(inps, list):
            inps = [inp for inp in inps]
        else:
            inps = inps
        output = m(inps)
        pred = output
        assert (pred.ndim == 4)
        pred = pred[:, eval_joints, :, :]
        if (output.shape[1] == 68):
            face_hand_num = 42
        else:
            face_hand_num = 110
        for i in range(output.shape[0]):
            bbox = bboxes[i].tolist()
            if combined_loss:
                (pose_coords_body_foot, pose_scores_body_foot) = heatmap_to_coord[0](pred[i][gt_val_dataset.EVAL_JOINTS[:(- face_hand_num)]], bbox, hm_shape=hm_size, norm_type=norm_type)
                (pose_coords_face_hand, pose_scores_face_hand) = heatmap_to_coord[1](pred[i][gt_val_dataset.EVAL_JOINTS[(- face_hand_num):]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_scores = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                (pose_coords, pose_scores) = heatmap_to_coord(pred[i][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)
            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape((- 1)).tolist()
            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float((np.mean(pose_scores) + (1.25 * np.max(pose_scores))))
            data['category_id'] = 1
            data['keypoints'] = keypoints
            kpt_json.append(data)
        bar.update()
    sysout = sys.stdout
    with open(os.path.join(opt.work_dir, 'test_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt.json'), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN), halpe=halpe)
    sys.stdout = sysout
    return res

def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    m = preset_model(cfg)
    # m = nn.DataParallel(m)
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    if combined_loss:
        criterion1 = builder.build_loss(cfg.LOSS.LOSS_1)
        criterion2 = builder.build_loss(cfg.LOSS.LOSS_2)
        criterion = [criterion1, criterion2]
    else:
        criterion = builder.build_loss(cfg.LOSS)
    if (cfg.TRAIN.OPTIMIZER == 'adam'):
        optimizer = jt.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif (cfg.TRAIN.OPTIMIZER == 'rmsprop'):
        optimizer = jt.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
    lr_scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)
    writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))
    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = train_dataset.set_attrs(batch_size=(cfg.TRAIN.BATCH_SIZE * num_gpu), shuffle=True, num_workers=opt.nThreads, drop_last=True,
                                           # keep_numpy_array=True,
                                           # buffer_size=1024 * 1024 * 1024,
                                           ) # tycoer
    # train_loader = jt.utils.data.DataLoader(train_dataset, batch_size=(cfg.TRAIN.BATCH_SIZE * num_gpu), shuffle=True, num_workers=opt.nThreads)
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    opt.trainIters = 0
    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['defaults']['lr']
        # current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')
        (loss, miou) = train(opt, train_loader, m, criterion, optimizer, writer)
        logger.epochInfo('Train', opt.epoch, loss, miou)
        lr_scheduler.step()
        if (((i + 1) % opt.snapshot) == 0):
            jt.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            with jt.no_grad():
                gt_AP = validate_gt(m.module, opt, cfg, heatmap_to_coord)
                rcnn_AP = validate(m.module, opt, heatmap_to_coord)
                logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} | rcnn mAP: {rcnn_AP} #####')
        if (i == cfg.TRAIN.DPG_MILESTONE):
            jt.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = jt.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
            train_loader = train_dataset.set_attrs(batch_size=(cfg.TRAIN.BATCH_SIZE * num_gpu), shuffle=True,
                                                   num_workers=opt.nThreads, drop_last=True)  # tycoer
            # train_loader = jt.utils.data.DataLoader(train_dataset, batch_size=(cfg.TRAIN.BATCH_SIZE * num_gpu), shuffle=True, num_workers=opt.nThreads)
    jt.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))

def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_parameters(jt.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = jt.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for (k, v) in pretrained_state.items() if ((k in model_state) and (v.shape == model_state[k].shape))}
        model_state.update(pretrained_state)
        model.load_parameters(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()
    return model

if (__name__ == '__main__'):
    main()
