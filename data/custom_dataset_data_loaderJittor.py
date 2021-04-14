
import jittor as jt
from jittor import init
from jittor import nn
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    from data.sym_datasetJittor import SymDataset
    dataset = SymDataset()
    print(('dataset [%s] was created' % dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = self.dataset.set_attrs(batch_size=opt.batchSize, shuffle=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

