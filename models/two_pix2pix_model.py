# two conditional GAN. First --> segmentation, second --> line detection
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .pix2pix_model import Pix2PixModel
from . import networks


class TwoPix2PixModel:
    def name(self):
        return 'TwoPix2PixModel'
    def initialize(self, opt):
        self.segmentation_GAN = Pix2PixModel()
        self.segmentation_GAN.initialize(opt)
        self.detection_GAN = Pix2PixModel()
        self.detection_GAN.initialize(opt)

    def set_input(self, input):
        """
        return {'A1': item1['A'], 'B1':item1['B'],
        'A2': item2['A'], 'B2': item2['B'],
        'A_paths1':item1['A_paths'], 'B_paths1':item1['B_paths'],
        'A_paths2':item2['A_paths'], 'B_paths2':item2['B_paths']}
        """
        input1 = input['dataset1_input']
        input2 = input['dataset2_input']
        self.segmentation_GAN.set_input(input1)
        self.detection_GAN.set_input(input2)

    def forward(self):
        self.segmentation_GAN.forward()
        self.detection_GAN.forward()
   
    def test(self):
        pass
    
    def get_image_paths(self):
        pass
    
    def backward_D(self):
        self.segmentation_GAN.backward_D()
        self.detection_GAN.backward_D()
    
    def backward_G(self):
        self.segmentation_GAN.backward_G()
        self.detection_GAN.backward_G()
    
    def optimize_parameters(self):
        # optimize parameter independently
        self.segmentation_GAN.optimize_parameters()
        self.detection_GAN.optimize_parameters()
    
    def get_current_errors(self):
        # @to output two errors
        error1 = self.segmentation_GAN.get_current_errors()
        error2 = self.detection_GAN.get_current_errors()
        return error1, error2
    
    def get_current_visuals(self):
        vis1 = self.segmentation_GAN.get_current_visuals()
        vis2 = self.detection_GAN.get_current_visuals()
        # @todo: only visualize detection result
        return vis2
    
    def save(self, label):
        label1 = 'seg_' + label
        label2 = 'dect_' + label
        self.segmentation_GAN.save(label1)
        self.detection_GAN.save(label2)

            
        