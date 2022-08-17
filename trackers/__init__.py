
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

def track(tracker, args, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores):
    hm = hm.numpy()
    online_targets = tracker.update(orig_img, inps, boxes, hm, cropped_boxes, im_name, scores, _debug=False)
    (new_boxes, new_scores, new_ids, new_hm, new_crop) = ([], [], [], [], [])
    for t in online_targets:
        tlbr = t.tlbr
        tid = t.track_id
        thm = t.pose
        tcrop = t.crop_box
        tscore = t.detscore
        new_boxes.append(tlbr)
        new_crop.append(tcrop)
        new_hm.append(thm)
        new_ids.append(tid)
        new_scores.append(tscore)
    new_hm = jt.Var(new_hm)
    return (new_boxes, new_scores, new_ids, new_hm, new_crop)
