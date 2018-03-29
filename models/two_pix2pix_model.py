# two conditional GAN. First --> segmentation, second --> line detection
import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .pix2pix_model import Pix2PixModel
from . import networks


class TwoPix2PixModel:
    def name(self):
        return 'TwoPix2PixModel'

    def initialize(self, opt):
        # copy from BaseModel
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # joint training or independent training
        if self.isTrain:
            self.isJointTrain = opt.joint_train != 0
        """
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        """
        
        if self.isTrain:            
            self.segmentation_GAN = Pix2PixModel()
            self.segmentation_GAN.initialize(opt)            
            self.detection_GAN = Pix2PixModel()
            self.detection_GAN.initialize(opt)
        else:
            self.seg_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,                
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            self.detec_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,                
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)            
            self.load_network(self.seg_netG, 'G', opt.which_epoch, 'seg')
            self.load_network(self.detec_netG, 'G', opt.which_epoch, 'detec')
            print('Warning: continue_train is not supported')

            print('---------- Networks initialized -------------')
            networks.print_network(self.seg_netG)
            networks.print_network(self.detec_netG)       
            print('-----------------------------------------------')        

    def set_input(self, input):        
        if self.isTrain:
            input1 = input['dataset1_input']
            input2 = input['dataset2_input']
            self.segmentation_GAN.set_input(input1)
            self.detection_GAN.set_input(input2)
        else:
            # same as in Pix2PixModel
            AtoB = self.opt.which_direction == 'AtoB'
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            if len(self.gpu_ids) > 0:
                input_A = input_A.cuda(self.gpu_ids[0], async=True)
                input_B = input_B.cuda(self.gpu_ids[0], async=True)
            self.input_A = input_A
            self.input_B = input_B
            self.image_paths = input['A_paths' if AtoB else 'B_paths']        

    def forward(self):
        if self.isJointTrain:
            pass
        else:
            self.segmentation_GAN.forward()
            self.detection_GAN.forward()
   
    def test(self):
        # forces outputs to not require gradients
        """
        self.real_A = Variable(self.input_A, volatile = True)
        self.fake_B = self.seg_netG(self.real_A)
        self.fake_C = self.detec_netG(self.real_A)            
        self.real_C = Variable(self.input_B, volatile = True)
        """
        self.real_A = Variable(self.input_A, volatile = True)
        self.fake_B = self.seg_netG(self.real_A)
        fake_B = (self.fake_B + 1.0)/2.0
        input_A = (self.real_A + 1.0)/2.0

        self.masked_A = (fake_B * input_A) * 2.0 - 1
        """
        fake_B = self.fake_B.data
        input_A = self.input_A   

        # composite image for detection GAN
        fake_B = (fake_B + 1.0)/2.0  # --> [0, 1]
        input_A = (input_A + 1.0)/2.0 # --> [0, 1]
        masked_A = torch.mul(input_A, fake_B)
        masked_A = masked_A * 2.0 - 1   # normalize to [-1, 1]

        masked_A = Variable(masked_A, volatile = True) # for debug
        self.masked_A = masked_A
        """
        self.fake_C = self.detec_netG(self.masked_A)
        self.real_C = Variable(self.input_B, volatile = True)      
    
    def get_image_paths(self):
        assert not self.isTrain
        return self.image_paths
    
    def backward_D(self):
        if self.isJointTrain:
            pass
        else:
            self.segmentation_GAN.backward_D()
            self.detection_GAN.backward_D()
    
    def backward_G(self):
        if self.isJointTrain:
            pass
        else:
            self.segmentation_GAN.backward_G()
            self.detection_GAN.backward_G()
    
    def optimize_parameters(self):
        if self.isJointTrain:
            pass
        else:
            # optimize parameter independently
            self.segmentation_GAN.optimize_parameters()
            self.detection_GAN.optimize_parameters()
    
    def get_current_errors(self):
        # @to output two errors
        error1 = self.segmentation_GAN.get_current_errors()
        error2 = self.detection_GAN.get_current_errors()
        return error1, error2
    
    def get_current_visuals(self):
        if self.isTrain:
            vis1 = self.segmentation_GAN.get_current_visuals()
            vis2 = self.detection_GAN.get_current_visuals()
            # @todo: only visualize detection result
            return vis2
        else:
            # same as in Pix2PixModel
            """
            self.real_A = Variable(self.input_A, volatile = True)
            self.fake_B = self.seg_netG(self.real_A)
            self.fake_C = self.detec_netG(self.real_A)            
            self.real_C = Variable(self.input_B, volatile = True)
            """
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            fake_C = util.tensor2im(self.fake_C.data)
            real_C = util.tensor2im(self.real_C.data)
            masked_A = util.tensor2im(self.masked_A.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('fake_C', fake_C), ('real_C', real_C),
            ('masked_A', masked_A)])
                 
            
    def save(self, label):
        label1 = 'seg_%s' % (label)
        label2 = 'detec_%s' % (label)
        self.segmentation_GAN.save(label1)
        self.detection_GAN.save(label2)
    
    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        self.segmentation_GAN.update_learning_rate()
        self.detection_GAN.update_learning_rate()
    
    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, phase_label):
        save_filename = '%s_%s_net_%s.pth' % (phase_label, epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))


            
        
