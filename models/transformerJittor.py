
import jittor as jt
from jittor import init
from jittor import nn
from .quatUtilsJittor import quat_conjugate, quat_rot_module

def rigidTsdf(points, trans, quat):
    p1 = translate_module(points, ((- 1) * trans))
    p2 = rotate_module(p1, quat)
    return p2

def planesymTransform(sample, plane):
    abc = plane[:, 0:3].unsqueeze(1).repeat(1, sample.shape[1], 1)
    d = plane[:, 3].unsqueeze(1).unsqueeze(1).repeat(1, sample.shape[1], 1)
    fenzi = (jt.sum((sample * abc), 2, True) + d)
    fenmu = (jt.norm(plane[:, 0:3], 2, 1, True).unsqueeze(1).repeat(1, sample.shape[1], 1) + 1e-05)
    x = (2 * jt.divide(fenzi, fenmu))
    y = jt.multiply(x.repeat(1, 1, 3), (abc / fenmu))
    return (sample - y)

def rotsymTransform(sample, quat):
    return rotate_module(sample, quat)

def rigidPointsTransform(points, trans, quat):
    quatConj = quat_conjugate(quat)
    p1 = rotate_module(points, quatConj)
    p2 = translate_module(p1, trans)
    return p2

def rotate_module(points, quat):
    nP = points.shape[1]
    quat_rep = quat.unsqueeze(1).repeat(1, nP, 1)
    zero_points = (0 * points[:, :, 0].clone().view(((- 1), nP, 1)))
    quat_points = jt.contrib.concat([zero_points, points], dim=2)
    rotated_points = quat_rot_module(quat_points, quat_rep)
    return rotated_points

def translate_module(points, trans):
    nP = points.shape[1]
    trans_rep = trans.repeat(1, nP, 1)
    return (points + trans_rep)

