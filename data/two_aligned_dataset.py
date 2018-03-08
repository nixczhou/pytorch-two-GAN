# two aligned dataset
from aligned_dataset import AlignedDataset

class TwoAlignedDataset:
    def initialize(self, opt):
        #self.opt = opt
        #self.root = opt.dataroot
        # set different phases (folders of image)
        opt1 = opt
        opt1.phase = opt.phase1
        opt1.dataset_model = 'aligned'
        opt2 = opt
        opt2.phase = opt.phase2
        opt2.dataset_model = 'aligned'
        self.dataset1 = AlignedDataset()
        self.dataset1.initialize(opt1)
        self.dataset2 = AlignedDataset()
        self.dataset2.initialize(opt2)

    def __getitem__(self, index):
        item1 = self.dataset1[index]
        item2 = self.dataset2[index]
        # Warning and to do
        # randomness in dataset1 will not be the same as in dataset2
        return {'dataset1_input':item1, 'dataset2_input':item2}
        #return {'A1': item1['A'], 'B1':item1['B'],
        #'A2': item2['A'], 'B2': item2['B'],
        #'A_paths1':item1['A_paths'], 'B_paths1':item1['B_paths'],
        #'A_paths2':item2['A_paths'], 'B_paths2':item2['B_paths']}
       
    def __len__(self):
        assert(len(self.dataset1) == len(self.dataset2))
        return len(self.dataset1)
    
    def name(self):
        return 'TwoAlignedDataset'

