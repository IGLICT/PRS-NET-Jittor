
import jittor as jt
from jittor import init
from jittor import nn
import os
import sys

class BaseModel(nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # self.Tensor = (torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor)
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def execute(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = ('%s_net_%s.pkl' % (epoch_label, network_label))
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path)


    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = ('%s_net_%s.pkl' % (epoch_label, network_label))
        if (not save_dir):
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if (not os.path.isfile(save_path)):
            print(('%s not exists yet!' % save_path))
        else:
            network.load(save_path)

    def update_learning_rate():
        pass

