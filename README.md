YOLOP_RESA
===



[TOC]



### 1, Introduce
-----
In this project, we aim to make a Lane_detection System for Self Driving Car with Instance Segmentation Approach. This system can detect not only weather each pixel is a part of Lanes but which Lane it belongs to.


### 2, Structure
-----
We use Instance Segmentaion as our main approach, We apply CNN to deal with our task. Our CNN Net consist of two part:
+ YOLOP net (CSPDarknet with SPP structure)
+ RESANET (RESA and BUSDecoder)

### 3, Training
-----
#### ▲Preparing
Our Training data need to be labelled in the form of Tusimple Dataset ,which means your dataset must consist of two part: your gt_image and json file.
#### ▲Step 1
With the labelled data, your need to put your dataset folder in ./data and create another three .json file -- test_label.json and valset.json
    
    data/your_data_folder 
        |-gt_image/ 
        |-JsonFile_label.json
        |-test_label.json
        |-valset.json
#### ▲Step 2 prepare testing set
then, cut (or copy ) some lines in JsonFile_label and paste them in test_label.json and valset.json
#### ▲Step 3 generating segmentation labels
>python tools/generate_seg_tusimple.py --root ./data/your_data_folder
#### ▲Step 4 modify config python file
1.set your training path

go to ./lib/config/resa_tusimple.py, find "dataset_path" and
"test_json_file" and change their content with './data/your_data_folder' and './data/your_data_folder/test_label.json' respectively.
#### ▲Step 5 training
Start training using following comment:
> python tools/train.py --view --cpu 0 --load_from your/pretrained/weight/path


### 4,testing
> python tools/train.py --view -- validate --cpu 0 --load_from your/pretrained/weight/path
---