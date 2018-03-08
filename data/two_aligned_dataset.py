# two aligned dataset
from aligned_dataset import AlignedDataset

class TwoAlignedDataset:
    def initialize(self, opt):
        #self.opt = opt
        #self.root = opt.dataroot
        # set different phases (folders of image)
        opt1 = opt
        opt1.phase = opt.phase1
        opt2 = opt
        opt2.phase = opt.phase2
        self.dataset1 = AlignedDataset(opt)
        self.dataset2 = AlignedDataset(opt)

    def __getitem__(self, index):
        item1 = self.dataset1[index]
        item2 = self.dataset2[index]
        # Warning and to do
        # randomness in dataset1 will not be the same as in dataset2
        return {'A1': item1['A'], 'B1':item1['B'],
        'A2': item2['A'], 'B2': item2['B']}
       
    def __len__(self):
        assert(len(self.dataset1) == len(self.dataset2))
        return len(self.dataset1)
    
    def name(self):
        return 'TwoAlignedDataset'

