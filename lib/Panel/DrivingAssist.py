import cv2
import numpy as np

class DrivingAssistant:
    def __init__(self,road_image):
        self.road_image = road_image
        self.h_sample = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    #def BirdEyeView(self)
    def CenterArrowedLine(self,left,right):
        #h_sample = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
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
        if self.h_sample[arr_end]>650:
            arr_end = self.h_sample.index(650)
        if self.h_sample[arr_start]<200:
            arr_start = self.h_sample.index(250)

        arr_line = [x for x in range(arr_start,arr_end)] 
        arr_len = len(arr_line)   
        #print("arr_end",arr_end)
        arr_start_x = (right[1][arr_start][0]+left[1][arr_start][0])/2
        arr_end_x = (right[1][arr_end][0]+left[1][arr_end][0])/2
        #arr_start_x是
        #x_last = 0
        #y_last = 0
        #for i in range(1,arr_len,10):
        #    y1 = int(self.h_sample[arr_line[i-1]])
        #    y2 = int(self.h_sample[arr_line[i]])
        #    x1 = int((right[1][i-1][0] + left[1][i-1][0])/2)
        #    x2 = int((right[1][i][0] + left[1][i][0])/2)
        #    x_last = x1
        #    y_last = y1
        #    cv2.line(self.road_image,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.arrowedLine(self.road_image,(arr_end_x,self.h_sample[arr_end_x]),(int(arr_start_x),self.h_sample[arr_start]),(255,255,255),3)
        #車道中間點
        #箭頭尾端
        arr_end_coor = (int(arr_end_x),self.h_sample[arr_end])
        cv2.circle(self.road_image,arr_end_coor,4,(255,0,0),-2)
        return arr_end_coor
    def KeepCenter(self,left,right,arr_end_coor):

        """
        Arguments:
        ----------
        left:The left lane coords
        right:The right lane coords
        arr_end_coor: the start of the arrow
        Return:
        ----------
        self.road_image:image that is added CenterPoint and KeepCenter Message
        """
        
        #把你的程式碼加在這裡
        center_coor = int(1280/2-1)
        cv2.circle(self.road_image, (center_coor,720-10), 5, (0,0,255), -2)
        cv2.line(self.road_image,(center_coor,710),(center_coor,650),(255,0,255),3)
        cv2.line(self.road_image,(center_coor,710),arr_end_coor,(0,0,255),2)
        if right[0] == -1 or left[0]==-1:  
            return self.road_image
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
            cv2.arrowedLine(self.road_image,(10,30),(70,30),(255,255,255),3,tipLength=0.5)
        elif flag is 'KeepLeft':
            cv2.arrowedLine(self.road_image,(70,30),(10,30),(255,255,255),3,tipLength=0.5)
        else:
            self.road_image = cv2.circle(self.road_image, (40,30), 25, (255,255,255), 2)
        cv2.putText(self.road_image, flag, (center_coor-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1, cv2.LINE_AA)
        #cv2.imshow('self.road_image',self.road_image)
        #cv2.waitKey(1)
        
        return self.road_image     