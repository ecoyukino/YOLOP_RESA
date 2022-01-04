import time
import torch
import numpy as np
from tqdm import tqdm
import pytorch_warmup as warmup
import cv2

from lib.RESANet.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from lib.dataset.datasets_resa import build_dataloader
from .recorder import build_recorder
from .net_utils import save_model, load_network
#user cdoe
import torchvision
import lib.utils_resa.transforms as tf

class DemoString:
    def __init__(self,path,Net,weight,device,cfg_resa):
        self.cfg = cfg_resa
        self.device = device
        self.recorder = build_recorder(self.cfg)
        self.path = path
        self.transformer = torchvision.transforms.Compose([
            tf.SampleResize((640, 384)),
            tf.GroupNormalize(mean=([103.939, 116.779, 123.68], (0, )), std=(
                [1., 1., 1.], (1, ))),
            ])
        self.net = Net
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.evaluator = build_evaluator(self.cfg)
        self.weight = weight
        self.decive = device
        self.net.load_state_dict(self.weight['net'],strict = False)
        self.net = self.net.to(self.device)
    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def run(self):
        # get frame:
        vidcap = cv2.VideoCapture(self.path)
        success,image = vidcap.read()
        print(image.shape)
        h,w,_ = image.shape
        size = (w,h)
        count = 0
        result_img = []
        
        while success:
            print("frame ",str(count+1))
            #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            img = self.transform(image)
            
            img = img.to(self.device)
            
            with torch.no_grad():
                output = self.net(img)[0]
                img = output['seg'].cpu().numpy()
                
                try:
                    result = self.evaluator.demo(self.path, output, img ,ori_image = image)
                except:
                    pass
                result_img.append(result)
                
                    
            success,image = vidcap.read()
            count += 1
        self.out_vid(result_img,size)
    def out_vid(self,img_list,size):
        import time,os
        a = time.localtime()
        re = time.strftime("%Y-%m-%d_%I_%M", a)
        model_dir = "./demo/{}".format(re)
        
        os.makedirs(model_dir)        
        file_name = self.path.split('/')[-1][:-4]
        out = cv2.VideoWriter("{}/{}.avi".format(model_dir,file_name),cv2.VideoWriter_fourcc(*'DIVX'),20,size)
        for i in range(len(img_list)):
            out.write(img_list[i])
        out.release()
    def transform(self,image):
        image = image[160:, :, :]
        image, = self.transformer((image,))
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image = image.unsqueeze(0)
        return image       