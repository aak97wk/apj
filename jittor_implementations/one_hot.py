import jittor as jt

def one_hot(labels, num_classes=1):
    assert labels.dtype == jt.int32 or labels.dtype == jt.int64, 'one_hot is only applicable to index tensor'
    assert labels.max() < num_classes, 'Class values must be smaller than num_classes.'
    index = jt.init.eye(num_classes, dtype=jt.int32)
    return index[labels]