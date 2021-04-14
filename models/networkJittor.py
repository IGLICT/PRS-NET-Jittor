
import jittor as jt
from jittor import init
from jittor import nn
from jittor import transform
from .transformerJittor import *
import scipy.io as sio
from .jittorAddon import *
jt.flags.use_cuda = 1

class Idn(nn.Module):

    def __init__(self, net):
        super(Idn, self).__init__()
        self.module = net

    def execute(self, inputs):
        return self.module(inputs)

def init_weights(net, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != (- 1)):
            init.gauss_(0.0, mean=gain)
        elif (classname.find('BatchNorm') != (- 1)):
            init.gauss_(1.0, mean=0.02)
            m.bias.data.fill_(0)

def init_net(net, init_gain=0.02, gpu_ids=[]):
    net = Idn(net)
    init_weights(net, gain=init_gain)
    return net

def define_PRSNet(input_nc, output_nc, conv_layers, num_plane, num_quat, biasTerms, useBn, activation, init_gain=0.02, gpu_ids=[]):
    if (activation == 'relu'):
        ac_fun = nn.relu(0)
    elif (activation == 'tanh'):
        ac_fun = nn.tanh()
    elif (activation == 'lrelu'):
        ac_fun = nn.LeakyReLU(scale=0.2)
    if useBn:
        print('using batch normalization')
    net = PRSNet(input_nc, output_nc, conv_layers, num_plane, num_quat, biasTerms, useBn, ac_fun)
    return init_net(net, init_gain, gpu_ids)

class PRSNet(nn.Module):

    def __init__(self, input_nc, output_nc, conv_layers, num_plane, num_quat, biasTerms, useBn=False, activation=nn.LeakyReLU(scale=0.2)):
        super(PRSNet, self).__init__()
        self.encoder = Encoder(input_nc, output_nc, conv_layers, useBn=useBn, activation=activation)
        self.pre = symPred((output_nc * (2 ** (conv_layers - 1))), num_plane, num_quat, biasTerms, activation=activation)

    def execute(self, voxel):
        return self.pre(self.encoder(voxel))

class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, conv_layers, useBn=False, activation=nn.LeakyReLU(scale=0.2)):
        super(Encoder, self).__init__()
        model = []
        output_nc = output_nc
        for i in range(conv_layers):
            model += [Conv3d(input_nc, output_nc, kernel_size=3, stride=1 ,padding=1)]
            model += [Pool3d(2), activation]
            input_nc = output_nc
            output_nc = (output_nc * 2)
        self.model = nn.Sequential(*model)

    def execute(self, input):
        return self.model(input)

class symPred(nn.Module):

    def __init__(self, input_nc, num_plane, num_quat, biasTerms, activation=nn.LeakyReLU(scale=0.2)):
        super(symPred, self).__init__()
        self.num_quat = num_quat
        for i in range(self.num_quat):
            quatLayer = [nn.Linear(input_nc, int((input_nc / 2))), activation, nn.Linear(int((input_nc / 2)), int((input_nc / 4))), activation]
            last = nn.Linear(int((input_nc / 4)), 4)
            last.bias.data = jt.transform.to_tensor(jt.array(biasTerms[('quat' + str((i + 1)))]))
            quatLayer += [last]
            setattr(self, ('quatLayer' + str((i + 1))), nn.Sequential(*quatLayer))
        self.num_plane = num_plane
        for i in range(self.num_plane):
            planeLayer = [nn.Linear(int(input_nc), int((input_nc / 2))), activation, nn.Linear(int((input_nc / 2)), int((input_nc / 4))), activation]
            last = nn.Linear(int((input_nc / 4)), 4)
            last.weight.data = jt.zeros((4, int(input_nc / 4)))
            last.bias.data = jt.transform.to_tensor(jt.array(biasTerms[('plane' + str((i + 1)))])).float()
            planeLayer += [last]
            setattr(self, ('planeLayer' + str((i + 1))), nn.Sequential(*planeLayer))

    def execute(self, feature):
        feature = feature.view((feature.shape[0], (- 1)))
        quat = []
        plane = []
        for i in range(self.num_quat):
            quatLayer = getattr(self, ('quatLayer' + str((i + 1))))
            quat += [normalize(quatLayer(feature))]
        for i in range(self.num_plane):
            planeLayer = getattr(self, ('planeLayer' + str((i + 1))))
            plane += [normalize(planeLayer(feature), 3)]
        return (quat, plane)

def normalize(x, enddim=4):
    x = (x / (1e-12 + jt.norm(x[:, :enddim], dim=1, k=2, keepdim=True)))
    return x

class RegularLoss(nn.Module):

    def __init__(self):
        super(RegularLoss, self).__init__()
        self.eye = jt.init.eye([3], dtype=jt.float32)

    def __call__(self, plane=None, quat=None, weight=1):
        reg_rot = jt.transform.to_tensor(jt.array([0]))
        reg_plane = jt.transform.to_tensor(jt.array([0]))
        if plane:
            p = [normalize(i[:, 0:3]).unsqueeze(2) for i in plane]
            x = jt.contrib.concat(p, dim=2)
            # y = jt.transpose(x, [1,2])
            y = jt.transpose(x, [0,2,1])
            reg_plane = ((jt.matmul(x, y) - self.eye).pow(2).sum(2).sum(1).mean() * weight)
        if quat:
            q = [i[:, 1:4].unsqueeze(2) for i in quat]
            x = jt.contrib.concat(q, dim=2)
            y = jt.transpose(x, [0,2,1])
            reg_rot = ((jt.matmul(x, y) - self.eye).pow(2).sum(2).sum(1).mean() * weight)
        return (reg_plane, reg_rot)

class symLoss(nn.Module):

    def __init__(self, gridBound, gridSize):
        super(symLoss, self).__init__()
        self.gridSize = gridSize
        self.gridBound = gridBound
        self.cal_distance = calDistence()

    def __call__(self, points, cp, voxel, plane=None, quat=None, weight=1):
        ref_loss = jt.transform.to_tensor(jt.array([0]))
        rot_loss = jt.transform.to_tensor(jt.array([0]))
        for p in plane:
            ref_points = planesymTransform(points, p)
            ref_loss += self.cal_distance(ref_points, cp, voxel, self.gridSize)
        for q in quat:
            rot_points = rotsymTransform(points, q)
            rot_loss += self.cal_distance(rot_points, cp, voxel, self.gridSize)
        return (ref_loss, rot_loss)

def pointClosestCellIndex(points, gridBound=0.5, gridSize=32):
    gridMin = ((- gridBound) + (gridBound / gridSize))
    gridMax = (gridBound - (gridBound / gridSize))
    inds = (((points - gridMin) * gridSize) / (2 * gridBound))
    inds = jt.round(jt.clamp(inds, min_v=0, max_v=gridSize-1))
    return inds

class calDistence(jt.Function):

    # @staticmethod
    def execute(self, trans_points, cp, voxel, gridSize, weight=1):
        if len(trans_points.shape) == 4:
            trans_points = trans_points.squeeze(dim=-1)
        nb = pointClosestCellIndex(trans_points)
        idx = jt.matmul(nb, jt.transform.to_tensor(jt.array([(gridSize ** 2), gridSize, 1])))
        mask = (1 - voxel.view((- 1), (gridSize ** 3)).gather(1, idx))
        idx = idx.unsqueeze(2)
        idx = idx.repeat(1, 1, 3)
        mask = mask.unsqueeze(2).repeat(1, 1, 3)
        closest_points = cp.gather(1, idx)
        self.constant = weight
        distance = (trans_points - closest_points)
        distance = (distance * mask)
        # self.save_for_backward(distance)
        self.saved_tensors = distance
        return (jt.mean(jt.sum(jt.sum(jt.pow(distance, 2), 2), 1)) * weight)

    # @staticmethod
    def grad(self, grad_output):
        distance = self.saved_tensors
        distance = distance[0]
        grad_trans_points = (((2 * distance) * self.constant) / distance.shape[0])
        return (grad_trans_points, None, None, None, None)

