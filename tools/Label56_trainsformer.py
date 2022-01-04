"""
Tusimple dataset label data with h_sample, a set of fixed y value, with the length of 56 and 
48. if the length of h_sample is 56, it starts from 160, or strats from 240. While in this code which
is implementing RESANET,only H_sample with length of 56 is available for validate while training, so we need 
to transform our label data from H_sample = 48 to 56.

--path is the path of existed json file  that record your labeled data which the len(H_sample) is 48
--edit is the path of a empty json file that you need to create by yourself

run this code with the cmd:
python tools/Label56_trainsformer.py --path [PATH] --edit [PATH]
All the code does is just convert your label json file from H_sample = 48 to 56 and record in a new 
json file you've just created (--edit)
"""

import json
import argparse
h_sample56 = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
left_pts = 56-48

json_path = "./data/vx3andlinco/JsonFile_label_old.json"
edit_path = "./data/vx3andlinco/JsonFile_label.json"
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='json_path ')
    parser.add_argument('--edit', type=str, help='json_path ')
    args = parser.parse_args()
    return args
def main():
    print("start transforming...")
    args = arg()
    json_path = args.path
    edit_path = args.edit
    with open(json_path, 'r') as f:
        with open(edit_path, 'w') as e:
            while True:
                data = f.readline()
                if data =="":
                    break
                dict_f = json.loads(data)
                dict_f["h_samples"] = h_sample56
                lanes = dict_f["lanes"]
                lane56 = [-2 for i in range(left_pts)]
            
                index = 0
                for i in range(len(lanes)):
                    lanes[i] = [*lane56,*lanes[i]]
                dict_f["lanes"] = lanes
                print("dict = ",dict_f)
                # write into new json file
                print("writing dict = ",dict_f)
                json.dump(dict_f,e)    
                e.write('\n')
            e.close()
        f.close()


if __name__ == '__main__':
    main()



