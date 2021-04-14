
import jittor as jt
from jittor import init
from jittor import nn
from jittor import transform
import numpy as np
# inds = torch.LongTensor([0, (- 1), (- 2), (- 3), 1, 0, 3, (- 2), 2, (- 3), 0, 1, 3, 2, (- 1), 0]).view((4, 4))
inds = jt.transform.to_tensor(jt.array([0, (- 1), (- 2), (- 3), 1, 0, 3, (- 2), 2, (- 3), 0, 1, 3, 2, (- 1), 0])).view((4,4))

def hamilton_product(q1, q2):
    q_size = q1.shape
    q1_q2_prods = []
    for i in range(4):
        q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
        q2_permute_0 = (q2_permute_0 * np.sign((inds[i][0] + 0.01)))
        q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
        q2_permute_1 = (q2_permute_1 * np.sign((inds[i][1] + 0.01)))
        q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
        q2_permute_2 = (q2_permute_2 * np.sign((inds[i][2] + 0.01)))
        q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
        q2_permute_3 = (q2_permute_3 * np.sign((inds[i][3] + 0.01)))
        q2_permute = jt.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)
        # q1q2_v1 = (q1 * q2_permute).sum(dim=2, keepdim=True)
        if (len(q1.shape) != len(q2_permute.shape)):
            q1q2_v1 = (q1.unsqueeze(-1) * q2_permute).sum(dim=2, keepdims=True)
        else:
            q1q2_v1 = (q1 * q2_permute).sum(dim=2, keepdims=True)

        q1_q2_prods.append(q1q2_v1)
    q_ham = jt.contrib.concat(q1_q2_prods, dim=2)
    return q_ham

def quat_conjugate(quat):
    q0 = quat[:, :, 0]
    q1 = ((- 1) * quat[:, :, 1])
    q2 = ((- 1) * quat[:, :, 2])
    q3 = ((- 1) * quat[:, :, 3])
    q_conj = jt.stack([q0, q1, q2, q3], dim=2)
    return q_conj

def quat_rot_module(points, quats):
    quatConjugate = quat_conjugate(quats)
    mult = hamilton_product(quats, points)
    mult = hamilton_product(mult, quatConjugate)
    return mult[:, :, 1:4]

