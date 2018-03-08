# two conditional GAN. First --> segmentation, second --> line detection
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .pix2pix_model import Pix2PixModel
from . import networks


class MaskPix2PixModel:
    def name(self):
        return 'MaskPix2PixModel'
    def initialize(self, opt):
        self.segmentation_GAN = Pix2PixModel()
        self.segmentation_GAN.initialize(opt)
        self.detection_GAN = Pix2PixModel()
        self.detection_GAN.initialize(opt)

    def set_input(self, input1, input2):
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
        pass
    
    def get_current_errors(self):
        pass
    
    def get_current_visuals(self):
        pass
    
    def save(self, label):
        pass
        