
import jittor as jt
from jittor import init
from jittor import nn
import os.path
from data.base_dataset import BaseDataset
import scipy.io as sio

def is_mat_file(filename):
    return filename.endswith('mat')

def make_dataset(dir):
    data = []
    assert os.path.isdir(dir), ('%s is not a valid directory' % dir)
    for (root, _, fnames) in sorted(os.walk(dir)):
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                data.append(path)
    return data

class SymDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_train = os.path.join(self.root, opt.phase)
        self.train_paths = sorted(make_dataset(self.dir_train))
        self.dataset_size = len(self.train_paths)

        self.set_attrs(total_len=self.dataset_size)

    def __getitem__(self, index):
        index = (index % self.dataset_size)
        data_path = self.train_paths[index]
        try:
            data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        except Exception as e:
            print(data_path, e)
            return None
        sample = data['surfaceSamples']
        voxel = data['Volume']
        cp = data['closestPoints']
        voxel = jt.transform.to_tensor(jt.array(voxel)).float().unsqueeze(0)
        sample = jt.transform.to_tensor(jt.array(sample)).float().t()
        cp = jt.transform.to_tensor(jt.array(cp)).float().view(((- 1), 3))
        input_dict = {'voxel': voxel, 'sample': sample, 'cp': cp, 'path': data_path}
        return input_dict

    def __len__(self):
        return ((self.dataset_size // self.opt.batchSize) * self.opt.batchSize)

    def name(self):
        return 'SymDataset'

