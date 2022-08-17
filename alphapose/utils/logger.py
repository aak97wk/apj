
import jittor as jt
from jittor import init
from jittor import nn

def board_writing(writer, loss, acc, iterations, dataset='Train'):
    writer.add_scalar('{}/Loss'.format(dataset), loss, iterations)
    writer.add_scalar('{}/acc'.format(dataset), acc, iterations)

def debug_writing(writer, outputs, labels, inputs, iterations):
    tmp_tar = jt.unsqueeze(labels[0], dim=1)
    tmp_inp = inputs[0]
    tmp_inp[0] += 0.406
    tmp_inp[1] += 0.457
    tmp_inp[2] += 0.48
    tmp_inp[0] += jt.sum(nn.interpolate(tmp_tar, scale_factor=4, mode='nearest'), dim=0)[0]
    tmp_inp.clamp_(0, 1)
    writer.add_image('Data/input', tmp_inp, iterations)
