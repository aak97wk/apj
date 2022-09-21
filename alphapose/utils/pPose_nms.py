import json
import os
import zipfile
import time
# from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
import numpy as np
' Constant Configuration '
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
alpha = 0.1
vis_thr = 0.2
oks_thr = 0.9
face_factor = 1.9
hand_factor = 0.55
hand_weight_score = 0.1
face_weight_score = 1.0
hand_weight_dist = 1.5
face_weight_dist = 1.0

def oks_pose_nms(data, soft=False):
    kpts = defaultdict(list)
    post_data = []
    for item in data:
        img_id = item['image_id']
        kpts[img_id].append(item)
    for (img_id, img_res) in kpts.items():
        for n_p in img_res:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            kpt = np.array(n_p['keypoints']).reshape((- 1), 3)
            for n_np in range(kpt.shape[0]):
                t_s = kpt[n_np][2]
                if (t_s > vis_thr):
                    kpt_score += t_s
                    valid_num += 1
            if (valid_num != 0):
                kpt_score = (kpt_score / valid_num)
            n_p['score'] = (kpt_score * box_score)
        if soft:
            keep = soft_oks_nms([img_res[i] for i in range(len(img_res))], oks_thr)
        else:
            keep = oks_nms([img_res[i] for i in range(len(img_res))], oks_thr)
        if (len(keep) == 0):
            post_data += img_res
        else:
            post_data += [img_res[_keep] for _keep in keep]
    return post_data

def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None):
    'OKS NMS implementations.\n    Args:\n        kpts_db: keypoints.\n        thr: Retain overlap < thr.\n        sigmas: standard deviation of keypoint labelling.\n        vis_thr: threshold of the keypoint visibility.\n    Returns:\n        np.ndarray: indexes to keep.\n    '
    if (len(kpts_db) == 0):
        return []
    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'] for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])
    order = scores.argsort()[::(- 1)]
    keep = []
    while (len(order) > 0):
        i = order[0]
        keep.append(i)
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, vis_thr)
        inds = np.where((oks_ovr <= thr))[0]
        order = order[(inds + 1)]
    keep = np.array(keep)
    return keep

def soft_oks_nms(kpts_db, thr, max_dets=20, sigmas=None, vis_thr=None):
    'Soft OKS NMS implementations.\n    Args:\n        kpts_db\n        thr: retain oks overlap < thr.\n        max_dets: max number of detections to keep.\n        sigmas: Keypoint labelling uncertainty.\n    Returns:\n        np.ndarray: indexes to keep.\n    '
    if (len(kpts_db) == 0):
        return []
    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])
    order = scores.argsort()[::(- 1)]
    scores = scores[order]
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while ((len(order) > 0) and (keep_cnt < max_dets)):
        i = order[0]
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, vis_thr)
        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)
        tmp = scores.argsort()[::(- 1)]
        order = order[tmp]
        scores = scores[tmp]
        keep[keep_cnt] = i
        keep_cnt += 1
    keep = keep[:keep_cnt]
    return keep

def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    'Calculate oks ious.\n    Args:\n        g: Ground truth keypoints.\n        d: Detected keypoints.\n        a_g: Area of the ground truth object.\n        a_d: Area of the detected object.\n        sigmas: standard deviation of keypoint labelling.\n        vis_thr: threshold of the keypoint visibility.\n    Returns:\n        list: The oks ious.\n    '
    if (sigmas is None):
        if (len(g) == 408):
            sigmas = (np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.8, 0.8, 0.8, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) / 10.0)
        elif (len(g) == 399):
            sigmas = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.04, 0.035, 0.031, 0.025, 0.02, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.01, 0.017, 0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.01, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.01, 0.008, 0.009, 0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031])
        elif (len(g) == 78):
            sigmas = (np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.8, 0.8, 0.8, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89]) / 10.0)
        else:
            sigmas = (np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0)
    vars = ((sigmas * 2) ** 2)
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d))
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = (xd - xg)
        dy = (yd - yg)
        e = (((((dx ** 2) + (dy ** 2)) / vars) / (((a_g + a_d[n_d]) / 2) + np.spacing(1))) / 2)
        if (vis_thr is not None):
            ind = (list((vg > vis_thr)) and list((vd > vis_thr)))
            e = e[ind]
        ious[n_d] = ((np.sum(np.exp((- e))) / len(e)) if (len(e) != 0) else 0.0)
    return ious

def _rescore(overlap, scores, thr, type='gaussian'):
    "Rescoring mechanism gaussian or linear.\n    Args:\n        overlap: calculated ious\n        scores: target scores.\n        thr: retain oks overlap < thr.\n        type: 'gaussian' or 'linear'\n    Returns:\n        np.ndarray: indexes to keep\n    "
    assert (len(overlap) == len(scores))
    assert (type in ['gaussian', 'linear'])
    if (type == 'linear'):
        inds = np.where((overlap >= thr))[0]
        scores[inds] = (scores[inds] * (1 - overlap[inds]))
    else:
        scores = (scores * np.exp(((- (overlap ** 2)) / thr)))
    return scores

def pose_nms(bboxes, bbox_scores, bbox_ids, pose_preds, pose_scores, areaThres=0, use_heatmap_loss=True):
    if ((pose_preds.shape[1] == 136) or (pose_preds.shape[1] == 133)):
        if (not use_heatmap_loss):
            global delta1, mu, delta2, gamma, scoreThreds, matchThreds, alpha
            delta1 = 1.0
            mu = 1.65
            delta2 = 8.0
            gamma = 3.6
            scoreThreds = 0.01
            matchThreds = 3.0
            alpha = 0.15
        return pose_nms_fullbody(bboxes, bbox_scores, bbox_ids, pose_preds, pose_scores, areaThres)
    else:
        return pose_nms_body(bboxes, bbox_scores, bbox_ids, pose_preds, pose_scores, areaThres)

def pose_nms_body(bboxes: np.ndarray,
                  bbox_scores: np.ndarray,
                  bbox_ids: np.ndarray,
                  pose_preds: np.ndarray,
                  pose_scores: np.ndarray,
                  areaThres=0):
    pose_scores[(pose_scores == 0)] = 1e-05
    kp_nums = pose_preds.shape[1]
    (res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids) = ([], [], [], [], [], [])

    # tycoer
    ori_bboxes = bboxes.copy()
    ori_bbox_scores = bbox_scores.copy()
    ori_bbox_ids = bbox_ids.copy()
    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]
    widths = (xmax - xmin)
    heights = (ymax - ymin)
    ref_dists = (alpha * np.maximum(widths, heights))
    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(axis=1)
    human_ids = np.arange(nsamples)
    mask = np.ones(len(human_ids)).astype(bool)

    # # tycoer
    # mask, human_ids, human_scores = np.array(mask), np.array(human_ids), np.array(human_scores)

    pick = []
    merge_ids = []
    while mask.any():
        tensor_mask = (mask == True)
        pick_id = np.argmax(human_scores[tensor_mask], axis=0).item()
        pick.append(human_ids[mask][pick_id].item()) # tycoer
        ref_dist = ref_dists[human_ids[mask][pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds[tensor_mask], pose_scores[tensor_mask], float(ref_dist))
        num_match_keypoints = PCK_match(pose_preds[tensor_mask][pick_id], pose_preds[tensor_mask], float(ref_dist))

        delete_ids = (np.arange(human_scores[tensor_mask].shape[0]))[((simi > gamma) | (num_match_keypoints >= matchThreds))]
        if (delete_ids.shape[0] == 0):
            delete_ids = pick_id
        merge_ids.append(human_ids[mask][delete_ids])
        newmask = mask [mask]
        newmask[delete_ids] = False
        mask[mask] = newmask

    assert (len(merge_ids) == len(pick))
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bboxes_pick = ori_bboxes[pick]
    bbox_ids_pick = ori_bbox_ids[pick]
    for j in range(len(pick)):
        ids = np.arange(kp_nums)
        max_score = np.max(scores_pick[(j, ids, 0)])
        if (max_score < scoreThreds):
            continue
        merge_id = merge_ids[j]
        (merge_pose, merge_score) = p_merge_fast(preds_pick[j],
                                                 ori_pose_preds[merge_id],
                                                 ori_pose_scores[merge_id],
                                                 float(ref_dists[pick[j]]))

        max_score = np.max(merge_score[ids])
        if (max_score < scoreThreds):
            continue
        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])
        bbox = bboxes_pick[j].tolist()
        bbox_score = bbox_scores_pick[j]
        if ((((1.5 ** 2) * (xmax - xmin)) * (ymax - ymin)) < areaThres):
            continue
        res_bboxes.append(bbox)
        res_bbox_scores.append(bbox_score)
        res_bbox_ids.append(ori_bbox_ids[merge_id].tolist())
        res_pose_preds.append(merge_pose)
        res_pose_scores.append(merge_score)
        res_pick_ids.append(pick[j])
    return (res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids)

def pose_nms_fullbody(bboxes: np.ndarray,
                      bbox_scores: np.ndarray,
                      bbox_ids: np.ndarray,
                      pose_preds: np.ndarray,
                      pose_scores: np.ndarray,
                      areaThres=0):
    '\n    Parametric Pose NMS algorithm\n    bboxes:         bbox locations list (n, 4)\n    bbox_scores:    bbox scores list (n, 1)\n    bbox_ids:       bbox tracking ids list (n, 1)\n    pose_preds:     pose locations list (n, kp_num, 2)\n    pose_scores:    pose scores list    (n, kp_num, 1)\n    '
    pose_scores[(pose_scores == 0)] = 1e-05
    kp_nums = pose_preds.shape[1]
    (res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids) = ([], [], [], [], [], [])

    ori_bboxes = bboxes.copy()
    ori_bbox_scores = bbox_scores.copy()
    ori_bbox_ids = bbox_ids.copy()
    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]
    widths = (xmax - xmin)
    heights = (ymax - ymin)
    ref_dists = (alpha * np.maximum(widths, heights))
    nsamples = bboxes.shape[0]
    human_scores = pose_scores[:, :, :].mean(axis=1)
    human_ids = np.arange(nsamples)
    mask = np.ones(len(human_ids)).astype(bool)
    pick = []
    merge_ids = []
    while mask.any():
        tensor_mask = (mask == True)
        pick_id = np.argmax(human_scores[tensor_mask])
        pick.append(human_ids[mask][pick_id])
        ref_dist = ref_dists[human_ids[mask][pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds[:, :, :][tensor_mask], pose_scores[:, :, :][tensor_mask], ref_dist, use_dist_mask=True)
        num_match_keypoints = PCK_match_fullbody(pose_preds[:, :, :][tensor_mask][pick_id], pose_scores[:, :, :][tensor_mask][pick_id], pose_preds[:, :, :][tensor_mask], ref_dist)
        delete_ids = np.array(np.arange(human_scores[tensor_mask].shape[0]))[((simi > gamma) | (num_match_keypoints >= matchThreds))]
        if (delete_ids.shape[0] == 0):
            delete_ids = pick_id
        merge_ids.append(human_ids[mask][delete_ids])
        newmask = mask[mask]
        newmask[delete_ids] = False
        mask[mask] = newmask
    assert (len(merge_ids) == len(pick))
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bboxes_pick = ori_bboxes[pick]
    bbox_ids_pick = ori_bbox_ids[pick]
    for j in range(len(pick)):
        ids = np.arange(kp_nums)
        max_score = np.max(scores_pick[(j, ids, 0)])
        if (max_score < scoreThreds):
            continue
        merge_id = merge_ids[j]
        (merge_pose, merge_score) = p_merge_fast(preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])
        max_score = np.max(merge_score[ids])
        if (max_score < scoreThreds):
            continue
        xmax = max(merge_pose[(ids, 0)])
        xmin = min(merge_pose[(ids, 0)])
        ymax = max(merge_pose[(ids, 1)])
        ymin = min(merge_pose[(ids, 1)])
        bbox = bboxes_pick[j].tolist()
        bbox_score = bbox_scores_pick[j]
        if ((((1.5 ** 2) * (xmax - xmin)) * (ymax - ymin)) < areaThres):
            continue
        res_bboxes.append(bbox)
        res_bbox_scores.append(bbox_score)
        res_bbox_ids.append(ori_bbox_ids[merge_id].tolist())
        res_pose_preds.append(merge_pose)
        res_pose_scores.append(merge_score)
        res_pick_ids.append(pick[j])
    return (res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids)

def filter_result(args):
    (score_pick, merge_id, pred_pick, pick, bbox_score_pick) = args
    global ori_pose_preds, ori_pose_scores, ref_dists
    kp_nums = ori_pose_preds.shape[1]
    ids = np.arange(kp_nums)
    max_score = np.max(score_pick[(ids, 0)])
    if (max_score < scoreThreds):
        return None
    (merge_pose, merge_score) = p_merge_fast(pred_pick, ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick])
    max_score = np.max(merge_score[ids])
    if (max_score < scoreThreds):
        return None
    xmax = max(merge_pose[:, 0])
    xmin = min(merge_pose[:, 0])
    ymax = max(merge_pose[:, 1])
    ymin = min(merge_pose[:, 1])
    if ((((1.5 ** 2) * (xmax - xmin)) * (ymax - ymin)) < (40 * 40.5)):
        return None
    return {'keypoints': (merge_pose - 0.3), 'kp_score': merge_score, 'proposal_score': ((np.mean(merge_score) + bbox_score_pick) + (1.25 * max(merge_score)))}

def p_merge(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '\n    Score-weighted pose merging\n    INPUT:\n        ref_pose:       reference pose          -- [kp_num, 2]\n        cluster_preds:  redundant poses         -- [n, kp_num, 2]\n        cluster_scores: redundant poses score   -- [n, kp_num, 1]\n        ref_dist:       reference scale         -- Constant\n    OUTPUT:\n        final_pose:     merged pose             -- [kp_num, 2]\n        final_score:    merged score            -- [kp_num]\n    '
    dist = np.sqrt(np.sum(axis=2))
    kp_num = ref_pose.shape[0]
    ref_dist = min(ref_dist, 15)
    mask = (dist <= ref_dist)
    final_pose = np.zeros((kp_num, 2))
    final_score = np.zeros(kp_num)
    if (cluster_preds.ndim == 2):
        cluster_preds.unsqueeze(0)
        cluster_scores.unsqueeze(0)
    if (mask.ndim == 1):
        mask.unsqueeze(0)
    for i in range(kp_num):
        cluster_joint_scores = cluster_scores[:, i][mask[:, i]]
        cluster_joint_location = cluster_preds[:, i, :][mask[:, i].unsqueeze((- 1)).repeat(1, 2)].view((np.sum(mask[:, i]), (- 1)))
        normed_scores = (cluster_joint_scores / np.sum(cluster_joint_scores))
        final_pose[(i, 0)] = np.dot(cluster_joint_location[:, 0], normed_scores.squeeze((- 1)))
        final_pose[(i, 1)] = np.dot(cluster_joint_location[:, 1], normed_scores.squeeze((- 1)))
        final_score[i] = np.dot(cluster_joint_scores.transpose(0, 1).squeeze(0), normed_scores.squeeze((- 1)))
    return (final_pose, final_score)

def p_merge_fast(ref_pose: np.ndarray,
                 cluster_preds: np.ndarray,
                 cluster_scores: np.ndarray,
                 ref_dist: float):
    '\n    Score-weighted pose merging\n    INPUT:\n        ref_pose:       reference pose          -- [kp_num, 2]\n        cluster_preds:  redundant poses         -- [n, kp_num, 2]\n        cluster_scores: redundant poses score   -- [n, kp_num, 1]\n        ref_dist:       reference scale         -- Constant\n    OUTPUT:\n        final_pose:     merged pose             -- [kp_num, 2]\n        final_score:    merged score            -- [kp_num]\n    '
    dist = np.sqrt(np.sum(np.power(ref_pose[np.newaxis, :] - cluster_preds, 2), axis=2))
    kp_num = ref_pose.shape[0]
    ref_dist = min(ref_dist, 15)
    mask = (dist <= ref_dist)
    final_pose = np.zeros((kp_num, 2))
    final_score = np.zeros(kp_num)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[None, ...]
        cluster_scores = cluster_scores[None, ...]
    if (mask.ndim == 1):
        mask = mask[..., None]
    masked_scores = cluster_scores * (mask[..., None])
    normed_scores = (masked_scores / np.sum(masked_scores, axis=0))
    final_pose = (cluster_preds * normed_scores.repeat(2, 2)).sum(axis=0)
    final_score = (masked_scores * normed_scores).sum(axis=0)
    return (final_pose, final_score)

def get_parametric_distance(i: int,
                            all_preds: np.ndarray,
                            keypoint_scores: np.ndarray,
                            ref_dist: float,
                            use_dist_mask: bool=False) -> np.ndarray:
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(np.power(pick_preds[None, :] - all_preds, 2), axis=2))
    mask = (dist <= 1)
    kp_nums = all_preds.shape[1]
    if use_dist_mask:
        dist_mask = (keypoint_scores.reshape(((- 1), kp_nums)) < scoreThreds)
        mask = (mask * dist_mask)
    score_dists = np.zeros((all_preds.shape[0], kp_nums))
    keypoint_scores = keypoint_scores.squeeze()
    if (keypoint_scores.ndim == 1):
        keypoint_scores = keypoint_scores[None]
    if (pred_scores.ndim == 1):
        pred_scores = pred_scores[:, None]
    pred_scores = pred_scores.repeat(all_preds.shape[0], 1).transpose(1, 0)
    score_dists[mask] = (np.tanh((pred_scores[mask] / delta1)) * np.tanh((keypoint_scores[mask] / delta1)))
    point_dist = np.exp((((- 1) * dist) / delta2))
    if use_dist_mask:
        point_dist[:, (- 110):(- 42)] = np.exp((((- 1) * dist[:, (- 110):(- 42)]) / (delta2 * face_factor)))
        point_dist[:, (- 42):] = np.exp((((- 1) * dist[:, (- 42):]) / (delta2 * hand_factor)))
        point_dist[dist_mask] = 0
        final_dist = (((np.mean(score_dists[:, :(- 110)], axis=1) + (np.mean(score_dists[:, (- 110):(- 42)], axis=1) * face_weight_score)) + (np.mean(score_dists[:, (- 42):], axis=1) * hand_weight_score)) + (mu * ((np.mean(point_dist[:, :(- 110)], axis=1) + (np.mean(point_dist[:, (- 110):(- 42)], axis=1) * face_weight_dist)) + (np.mean(point_dist[:, (- 42):], axis=1) * hand_weight_dist))))
    else:
        final_dist = (np.sum(score_dists, axis=1) + (mu * np.sum(point_dist, axis=1)))
    return final_dist

def PCK_match(pick_pred: np.ndarray,
              all_preds: np.ndarray,
              ref_dist: float) -> np.ndarray:
    dist = np.sqrt(np.sum(np.power(pick_pred[np.newaxis, :] - all_preds, 2), axis=2))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = np.sum(((dist / ref_dist) <= 1), axis=1)
    return num_match_keypoints

def PCK_match_fullbody(pick_pred, pred_score, all_preds, ref_dist):
    kp_nums = pred_score.shape[0]
    mask = (pred_score.reshape((1, kp_nums, 1)).repeat(all_preds.shape[0], 1, 2) > (scoreThreds / 2)).float()
    if (mask.sum() < 2):
        return np.zeros(all_preds.shape[0])
    dist = np.sqrt(np.sum(np.power(pick_pred[np.newaxis, :] - all_preds, 2), axis=2))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints_body = np.sum(((dist[:, :26] / ref_dist) <= 1), axis=1)
    num_match_keypoints_face = np.sum(((dist[:, 26:94] / ref_dist) <= face_factor), axis=1)
    num_match_keypoints_hand = np.sum(((dist[:, 94:] / ref_dist) <= hand_factor), axis=1)
    num_match_keypoints = (((((num_match_keypoints_body + num_match_keypoints_face) + num_match_keypoints_hand) / mask.sum()) / 2) * kp_nums)
    return num_match_keypoints

def write_json(all_results, outputpath, form=None, for_eval=False, outputfile='alphapose-results.json'):
    '\n    all_result: result dict of predictions\n    outputpath: output directory\n    '
    json_results = []
    json_results_cmu = {}
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            if for_eval:
                result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[(- 1)])
            else:
                result['image_id'] = os.path.basename(im_name)
            result['category_id'] = 1
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[(n, 0)]))
                keypoints.append(float(kp_preds[(n, 1)]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            if ('box' in human.keys()):
                result['box'] = human['box']
            if ('idx' in human.keys()):
                result['idx'] = human['idx']
            if ('pred_xyz_nps' in human.keys()):
                pred_xyz_nps = human['pred_xyz_nps']
                pred_xyz_nps = pred_xyz_nps.numpy().tolist()
                result['pred_xyz_nps'] = pred_xyz_nps
            if (form == 'cmu'):
                if (result['image_id'] not in json_results_cmu.keys()):
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = 'AlphaPose v0.3'
                    json_results_cmu[result['image_id']]['bodies'] = []
                tmp = {'joints': []}
                result['keypoints'].append(((result['keypoints'][15] + result['keypoints'][18]) / 2))
                result['keypoints'].append(((result['keypoints'][16] + result['keypoints'][19]) / 2))
                result['keypoints'].append(((result['keypoints'][17] + result['keypoints'][20]) / 2))
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][(i + 1)])
                    tmp['joints'].append(result['keypoints'][(i + 2)])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif (form == 'open'):
                if (result['image_id'] not in json_results_cmu.keys()):
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = 'AlphaPose v0.3'
                    json_results_cmu[result['image_id']]['people'] = []
                tmp = {'pose_keypoints_2d': []}
                result['keypoints'].append(((result['keypoints'][15] + result['keypoints'][18]) / 2))
                result['keypoints'].append(((result['keypoints'][16] + result['keypoints'][19]) / 2))
                result['keypoints'].append(((result['keypoints'][17] + result['keypoints'][20]) / 2))
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27, 36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][(i + 1)])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][(i + 2)])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)
    if (form == 'cmu'):
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if (not os.path.exists(os.path.join(outputpath, 'sep-json'))):
                os.mkdir(os.path.join(outputpath, 'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath, 'sep-json', (name.split('.')[0] + '.json')), 'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif (form == 'open'):
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if (not os.path.exists(os.path.join(outputpath, 'sep-json'))):
                os.mkdir(os.path.join(outputpath, 'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath, 'sep-json', (name.split('.')[0] + '.json')), 'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results))

def ppose_nms_validate_preprocess(_res):
    res = {}
    for data in _res:
        if (data['image_id'] not in res.keys()):
            res[data['image_id']] = []
        res[data['image_id']].append(data)
    _tmp_data = {}
    for key in res.keys():
        pose_coords = []
        pose_scores = []
        bboxes = []
        scores = []
        ids = []
        i = 0
        cur = res[key]
        for pose in cur:
            bboxes.append([pose['bbox'][0], pose['bbox'][1], (pose['bbox'][0] + pose['bbox'][2]), (pose['bbox'][1] + pose['bbox'][3])])
            kpts = np.array(pose['keypoints'], dtype=np.float32).reshape(((- 1), 3))
            coords = kpts[:, 0:2]
            p_scores = kpts[:, 2]
            s = ((pose['score'] - np.mean(p_scores)) - (1.25 * np.max(p_scores)))
            scores.append(s)
            pose_coords.append(np.from_numpy(coords).unsqueeze(0))
            pose_scores.append(np.from_numpy(p_scores).unsqueeze(0))
            ids.append(i)
            i += 1
        preds_img = np.contrib.concat(pose_coords)
        preds_scores = np.contrib.concat(pose_scores)[:, :, None]
        boxes = np.array(np.array(bboxes, dtype=np.float32))
        scores = np.array(np.array(scores, dtype=np.float32).reshape((- 1), 1))
        ids = np.array(np.array(ids, dtype=np.float32).reshape((- 1), 1))
        _tmp_data[key] = (boxes, scores, ids, preds_img, preds_scores)
    return _tmp_data
