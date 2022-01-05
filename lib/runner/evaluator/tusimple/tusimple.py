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
        img：原本的圖片
        coords 是預測出來的線條，他的格式和tusimple的lanes一樣
        """



        center_x = 1280/2
        leftlane_starndar = 1280/2
        coord_index = 0
        
        coords = self.sort_key(coords)


        #print("coords.len = ",len(coords))
        """
        這邊辨識哪兩條車道線屬於行駛中的車道
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
        left所記錄的是車子行駛中的車道的左側車道線 right則是行駛車道中的右側車道線
        紀錄方法：[車道線編號,車道的點]
        例如：[1, [[5, 160], [6, 170], [9, 180].......]]
        這是為了畫中間的箭頭
        """

        left = [] # [lane_index(int),lane(list)]
        right = [] # [lane_index(int),lane(list)]
        if len(coords)==0: # 沒有車道線被偵測出
            left.append(-1)
            right.append(-1)
        elif coord_index == 0: # 左邊沒有車道線
            left.append(-1)
            right.append(0)
            right.append(coords[0])
        elif coord_index == len(coords) and len(coords) != 0:# 右邊沒有車道線
            right.append(-1)
            left.append(len(coords)-1)
            left.append(coords[-1])
        else : # 左右都有車道線(只有這個才能畫箭頭)
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
        
        #這邊是畫點點或話線的地方
        for coord in coords:
            if lane_index == left[0] or lane_index ==right[0]:#如果這條線是車子行駛的車道 畫線 
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
            else:#不是車子行駛的車道 畫圈圈
                for x, y in coord:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 4, color[color_index], 2)    
            color_index += 1
            lane_index += 1
        
        """
        這裡是畫箭頭的程式，看你要不要順便把它改成彎曲的箭頭
        right[1] = 車道線的座標們
        right[1][i][0] = 第i個點的x座標
        right[1][i][1] = 第i個點的y座標
        第一個點的y座標是160 依序下去為170 180 ...跟H_sample一樣
        left同理

        看你要不要改這邊 這邊是車道上的箭頭

        很醜

        """       
        y_sample = [i for i in range(56)]
        arr_start = 0
        arr_end = 0
        for i in y_sample:
            if left[1][i][0] >0 and right[1][i][0] >0 :
                arr_start = i
                break
            
        #print("arr_start",arr_start)
        y_sample = [i for i in range(55,-1,-1)]
        #print(y_sample)
        for i in y_sample:
            if left[1][i][0] >0 and right[1][i][0] >0 :
                arr_end = i
                break
        #print("arr_end",arr_end)
        arr_start_x = (right[1][arr_start][0]+left[1][arr_start][0])/2
        arr_end_x = (right[1][arr_end][0]+left[1][arr_end][0])/2
        #arr_start_x是
        h_sample = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        cv2.arrowedLine(img,(int(arr_end_x),h_sample[arr_end]),(int(arr_start_x),h_sample[arr_start]),(255,255,255),3)

        #arr_end_x 是箭頭的最底下的部分的x座標 arr_start_x是箭頭尖端的x座標
        

        """
        if not os.path.exists(osp.dirname(file_path)):
            os.makedirs(osp.dirname(file_path))
            img_save_name = os.path.join(file_path, image_name)
            print("save dir = ",img_save_name)
        """
        img = self.KeepCenter(img,left,right)#把你的程式碼加在這裡


        #print(img_save_name)
        if type(image_path) == type("string"):
            cv2.imwrite(image_path,img)
        return img
    def KeepCenter(self,img,left,right):
        """
        你會用到的function(只須改中文的地方)
        right[1] = 車道線的座標們
        right[1][i][0] = 第i個點的x座標
        right[1][i][1] = 第i個點的y座標
        畫箭頭的程式碼
        cv2.arrowedLine(img,(箭頭的尾端座標)),(箭頭的尖端座標),(255,255,255),3)
        放字串的程式碼
        cv2.
        cv2.putText(
            img, "你要放的字串", (640,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1, cv2.LINE_AA
            )
        畫圈圈的程式碼(標示車子的中心點)
        cv2.circle(img, (圓心座標), 4, (顏色), 2) 
        改完後
        $git bash:
        $git add  lib/runner/evaluator/tusimple/tusimple.py
        $git commit -m "KeepCenter_Complete"
        then
        $git push
        然後再發pull request給我
        就降
        got it?

        cmd不認識git指令 我好想在學長的vscode上裝git的plugin
        """
        
        #把你的程式碼加在這裡
        center_coor = int(1280/2-1)
        img = cv2.circle(img, (center_coor,720-10), 4, (0,0,255), -2)
        if right[0] == -1 or left[0]==-1:  
            return img
        
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
                if  coord[i][0] >0 :# 有點
                    a+=1
            if a == len(coords):
                break
        
       
        coords.sort(key = lambda x: x[i][0] ) 
        #print("coords = ",coords)
        #print("coords len = ",len(coords))
        return coords
