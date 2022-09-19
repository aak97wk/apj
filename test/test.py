from jittor.models.resnet import resnet50
import jittor as jt

import numpy as np
if __name__ == '__main__':
    i = jt.array([0])
    var = jt.ones((1, 17, 2))
    res = var[i]
    print(res.shape)

    i = np.array([0])
    var = np.ones((1, 17, 2))
    res = var[i]
    print(res.shape)