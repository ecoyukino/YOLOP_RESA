import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

from lib.runner.runner import Runner
from lib.dataset.datasets_resa import build_dataloader
from lib.utils_resa.config import Config

from lib.runner.demo_runner import DemoString

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):
    model = get_net(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(model)
    checkpoint = torch.load(cfg.MODEL.PRETRAINED, map_location= device)
    model.load_state_dict(checkpoint['net'],strict = False)
    model = model.to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model
    cfg_resa = Config.fromfile('./lib/config/resa_tusimple.py')
    #cfg = Config.fromfile('./configs/tusimple.py')
    cfg_resa.gpus = len(opt.gpus)

    cfg_resa.load_from = opt.load_from
    cfg_resa.finetune_from = opt.finetune_from
    cfg_resa.view = opt.view

    cfg_resa.work_dirs = opt.work_dirs + '/' + 'TuSimple'

    cudnn.benchmark = True
    cudnn.fastest = True

    
   


   
    # Set Dataloader
    dir = "./demo/source"
    vid_list = os.listdir(dir)
    for path in vid_list:
        p = "./demo/source/{}".format(path)
        DemoRunner = DemoString(p,model,checkpoint,device,cfg_resa)
        DemoRunner.run()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=cfg.MODEL.PRETRAINED, help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=cfg.MODEL.PRETRAINED,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--view',
        action='store_true',
        help='whether to show visualization result')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('--demo',action='store_true',
        help='whether to demo the checkpoint during training')
   
    
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
