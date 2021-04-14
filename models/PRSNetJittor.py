
import jittor as jt
from jittor import init
from jittor import nn
from .base_modelJittor import BaseModel
from .networkJittor import *
import numpy as np
jt.flags.use_cuda = 1
class PRSNet(BaseModel):

    def name(self):
        return 'PRSNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        biasTerms = {}
        biasTerms['plane1'] = [1, 0, 0, 0]
        biasTerms['plane2'] = [0, 1, 0, 0]
        biasTerms['plane3'] = [0, 0, 1, 0]
        biasTerms['quat1'] = [0, 0, 0, np.sin((np.pi / 2))]
        biasTerms['quat2'] = [0, 0, np.sin((np.pi / 2)), 0]
        biasTerms['quat3'] = [0, np.sin((np.pi / 2)), 0, 0]
        if (opt.num_plane > 3):
            for i in range(4, (opt.num_plane + 1)):
                plane = np.random.random_sample((3,))
                biasTerms[('plane' + str(i))] = ((plane / np.linalg.norm(plane)).tolist() + [0])
        if (opt.num_quat > 3):
            for i in range(4, (opt.num_quat + 1)):
                quat = np.random.random_sample((4,))
                biasTerms[('quat' + str(i))] = (quat / np.linalg.norm(quat)).tolist()
        self.opt = opt
        self.netPRS = define_PRSNet(opt.input_nc, opt.output_nc, opt.conv_layers, opt.num_plane, opt.num_quat, biasTerms, opt.bn, opt.activation, gpu_ids=self.gpu_ids)
        if ((not self.isTrain) or opt.continue_train or opt.load_pretrain):
            pretrained_path = ('' if (not self.isTrain) else opt.load_pretrain)
            self.load_network(self.netPRS, self.name(), opt.which_epoch, pretrained_path)
        if self.isTrain:
            self.sym_loss = symLoss(opt.gridBound, opt.gridSize)
            self.reg_loss = RegularLoss()
            self.loss_names = ['ref', 'rot', 'reg_plane', 'reg_rot']
            params = list(self.netPRS.parameters())
            self.optimizer_PRS = jt.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def execute(self, voxel, points, cp):
        voxel = jt.Var(voxel)
        points = jt.Var(points)
        cp = jt.Var(cp)
        (quat, plane) = self.netPRS(voxel)
        (loss_ref, loss_rot) = self.sym_loss(points, cp, voxel, plane=plane, quat=quat)
        (loss_reg_plane, loss_reg_rot) = self.reg_loss(plane=plane, quat=quat, weight=self.opt.weight)
        return [loss_ref, loss_rot, loss_reg_plane, loss_reg_rot]

    def inference(self, voxel):
        if (len(self.gpu_ids) > 0):
            voxel = jt.Var(voxel)
        else:
            voxel = jt.Var(voxel)
        self.netPRS.eval()
        with jt.no_grad():
            (quat, plane) = self.netPRS(voxel)
        return (plane, quat)

    def save(self, which_epoch):
        self.save_network(self.netPRS, self.name(), which_epoch, self.gpu_ids)

class Inference(PRSNet):

    def execute(self, voxel):
        return self.inference(voxel)

