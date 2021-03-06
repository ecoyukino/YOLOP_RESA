import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.runner.logger import get_logger

from lib.runner.registry import EVALUATOR 
import json
import os
import cv2

from .lane import LaneEval

import os.path as osp
import numpy as np
#import cv2
import torchvision
import lib.utils_resa.transforms as tf
from lib.Panel.DrivingAssist import DrivingAssistant
from lib.Panel.LaneCorrect import LaneCorrector
def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


@EVALUATOR.register_module
class Tusimple(nn.Module):
    def __init__(self, cfg):
        super(Tusimple, self).__init__()
        self.cfg = cfg
        print("cfg = ",cfg) 
        exp_dir = os.path.join(self.cfg.work_dir, "output")
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        self.out_path = os.path.join(exp_dir, "coord_output")
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        self.dump_to_json = [] 
        self.thresh = cfg.evaluator.thresh
        self.logger = get_logger('resa')
        if cfg.view:
            self.view_dir = os.path.join(self.cfg.work_dir, 'vis')

    def evaluate_pred(self, dataset, seg_pred, exist_pred, batch):
        img_name = batch['meta']['img_name']
        img_path = batch['meta']['full_img_path']
        
        for b in range(len(seg_pred)):
            #print("seg.shape = ",seg_pred.shape)
            seg = seg_pred[b]
            
            exist = [1 if exist_pred[b, i] >
                     0.5 else 0 for i in range(self.cfg.num_classes-1)]
            lane_coords = dataset.probmap2lane(seg, exist, thresh = self.thresh)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(self.out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            if self.cfg.view:
                img = cv2.imread(img_path[b])
                new_img_name = img_name[b].replace('/', '_')
                save_dir = os.path.join(self.view_dir, new_img_name)
                dataset.view(img, lane_coords, save_dir)
    def demo_pred(self, path,seg_pred,exist_pred, batch, ori_img = 0):
        #print("demo_pred")
        try:
            img_path = batch['meta']
            img_path = img_path.replace("\\","/")
        #print("img_path = ",img_path)
            img_name = img_path.split('/')[-1]
        except :
            pass
        #print("img_name = ",img_name)
        for b in range(len(seg_pred)):
            #print("seg.shape = ",seg_pred.shape)
            seg = seg_pred[b]
            #print("---1---")
            exist = [1 if exist_pred[b, i] >
                     0.5 else 0 for i in range(self.cfg.num_classes-1)]
            #print("---2---")
            demo_data = TuSimple_Demo(path)
            lane_coords = demo_data.demo_probmap2lane(seg, exist, thresh = self.thresh)
            #print("---3---")

            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            #print("---4---")
            """
            
            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(self.out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)
            
            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['run_time'] = 0

            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            """
            
            #print("img_path = ",img_path)
            if type(ori_img) == type(int(0)):
                img = cv2.imread(img_path)
                new_img_name = img_name[b].replace('/', '_')
                save_dir = "./inference/ResaNet_output_vx3/"
                demo_data.view(img, lane_coords, img_path)
            else:
                img = ori_img
                img = demo_data.view(img, lane_coords, img)
                return img
    def evaluate(self, dataset, output, batch):
        seg_pred, exist_pred = output['seg'], output['exist']
        seg_pred = F.softmax(seg_pred, dim=1)
        
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        self.evaluate_pred(dataset, seg_pred, exist_pred, batch)
    def demo(self,path, output, batch,ori_image = 0):
        seg_pred, exist_pred = output['seg'], output['exist']
        seg_pred = F.softmax(seg_pred, dim=1)
        
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        if type(ori_image) == type(int(0)) :
            self.demo_pred(path,seg_pred,exist_pred, batch)
        else:
            img = self.demo_pred(path,seg_pred,exist_pred, batch, ori_img = ori_image)
            return img
    def summarize(self):
        best_acc = 0
        output_file = os.path.join(self.out_path, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_result, acc = LaneEval.bench_one_submit(output_file,
                            self.cfg.test_json_file)

        self.logger.info(eval_result)
        self.dump_to_json = []
        best_acc = max(acc, best_acc)
        return best_acc




class TuSimple_Demo():
    def __init__(self,path):
        #print("TuSimple_Demo")
        self.path = path

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end+1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i+1] < 0]
                gap_end = [i+1 for i,
                           x in enumerate(lane[:-1]) if x < 0 and lane[i+1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                gap_end[i] - id) / gap_width * lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end+1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_lane(self, prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)
    
        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = resize_shape
        H -= 160
    
        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)
        #print(coords.shape)

        return coords

    def demo_probmap2lane(self, seg_pred, exist, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold
    
        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """

        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []
        #print("prp1")
        for i in range(6):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
    
        #print("prp2")
        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        #print(coordinates)
        #print("prp3")
        return coordinates
    def view(self, img, coords,image_path = 0):
        """
        img??????????????????
        coords ??????????????????????????????????????????tusimple???lanes??????
        """

        Self_Correction_System = LaneCorrector(coords)#Salmon's code, add anonther arguments if you need
        coords = Self_Correction_System.Salmon_Fliter()# Salmon' code is written here

        center_x = 1280/2
        leftlane_starndar = 1280/2
        coord_index = 0
        
        coords = self.sort_key(coords)


        #print("coords.len = ",len(coords))
        """
        ??????????????????????????????????????????????????????
        """
        for coord in coords:
            total_pt_number = 0
            x_Sum = 0
            for i in range(56):
                if coord[i][0]>0 and coord[i][1]>=300:
                    total_pt_number+=1
                    x_Sum+= coord[i][0]
            if int(x_Sum/total_pt_number) <= leftlane_starndar:
                coord_index+=1
            else:
                break
        #print("LeftRight")
        
        """
        left????????????????????????????????????????????????????????? right???????????????????????????????????????
        ???????????????[???????????????,????????????]
        ?????????[1, [[5, 160], [6, 170], [9, 180].......]]
        ??????????????????????????????
        """

        left = [] # [lane_index(int),lane(list)]
        right = [] # [lane_index(int),lane(list)]
        if len(coords)==0: # ???????????????????????????
            left.append(-1)
            right.append(-1)
        elif coord_index == 0: # ?????????????????????
            left.append(-1)
            right.append(0)
            right.append(coords[0])
        elif coord_index == len(coords) and len(coords) != 0:# ?????????????????????
            right.append(-1)
            left.append(len(coords)-1)
            left.append(coords[-1])
        else : # ?????????????????????(???????????????????????????)
            right.append(coord_index)
            right.append(coords[coord_index])
            left.append(coord_index-1)
            left.append(coords[coord_index-1])
        #y_sam = [x for x in range(360,720,10)].reverse()
        #print("draw.....")
        color = [
            (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(255,255,0)
        ]
        color_index = 0
        lane_index = 0
        
        #????????????????????????????????????
        for coord in coords:
            if lane_index == left[0] or lane_index ==right[0]:#??????????????????????????????????????? ?????? 
                drawline = False
                for x, y in coord:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    if not drawline:
                        x1 = x
                        y1 = y
                        drawline = True
                    else:
                        x2 = x
                        y2 = y
                        cv2.line(img,(x1,y1),(x2,y2),color[color_index],5)
                        x1 = x2
                        y1 = y2
                    i+=1
            else:#??????????????????????????? ?????????
                for x, y in coord:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 4, color[color_index], 2)    
            color_index += 1
            lane_index += 1
        

        #arr_end_x ?????????????????????????????????x?????? arr_start_x??????????????????x??????
        assistant = DrivingAssistant(img)
        arr_end_coor = assistant.CenterArrowedLine(left,right)
        assistant.KeepCenter(left,right,arr_end_coor)



        #print(img_save_name)
        if type(image_path) == type("string"):
            cv2.imwrite(image_path,img)
        return assistant.road_image
    def SelfCorrection(self,coords):
        """
        Arguments:
        ----------
        coords:coords of lane that haven't been corrected
        Return:
        ----------
        coords:coords of lane that have been corrected
        """
        return coords
    def KeepCenter(self,img,left,right,arr_end_coor):
        """
        Arguments:
        ----------
        left:The left lane coords
        right:The right lane coords
        arr_end_coor: the start of the arrow
        Return:
        ----------
        img:image that is added CenterPoint and KeepCenter Message
        """
        
        #??????????????????????????????
        center_coor = int(1280/2-1)
        cv2.circle(img, (center_coor,720-10), 5, (0,0,255), -2)
        cv2.line(img,(center_coor,710),(center_coor,650),(255,0,255),3)
        cv2.line(img,(center_coor,710),arr_end_coor,(0,0,255),2)
        if right[0] == -1 or left[0]==-1:  
            return img
        """
        right = np.array(right[1])
        left = np.array(left[1])
        
        idx_r = np.where(right[:,0]>-1)[0][-1]
        idx_l = np.where(left[:,0]>-1)[0][-1]
        if idx_r > idx_l:
            idx_r = idx_l
        else:
            idx_l = idx_r
        lane_center  = right[idx_r,0] +  left[idx_l,0]
        
        lane_center = lane_center//2
        
        if lane_center - center_coor > 10:
            flag = 'KeepRight'
        elif lane_center - center_coor < -10:
            flag = 'KeepLeft'
        else:
            flag = 'In the center'
        """
        lane_center = arr_end_coor[0]
        if lane_center - center_coor > 10:
            flag = 'KeepRight'
        elif lane_center - center_coor < -10:
            flag = 'KeepLeft'
        else:
            flag = 'In the center'

        #flag is KeepCenter Message
        if flag is 'KeepRight':
            cv2.arrowedLine(img,(10,30),(70,30),(255,255,255),3,tipLength=0.5)
        elif flag is 'KeepLeft':
            cv2.arrowedLine(img,(70,30),(10,30),(255,255,255),3,tipLength=0.5)
        else:
            img = cv2.circle(img, (40,30), 25, (255,255,255), 2)
        cv2.putText(img, flag, (center_coor-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1, cv2.LINE_AA)
        #cv2.imshow('img',img)
        #cv2.waitKey(1)

        return img    
    def sort_key(self, coords):
        
        for i in range(56):
            a = 0
            for coord in coords:
                if  coord[i][0] >0 :# ??????
                    a+=1
            if a == len(coords):
                break
        
       
        coords.sort(key = lambda x: x[i][0] ) 
        #print("coords = ",coords)
        #print("coords len = ",len(coords))
        return coords
