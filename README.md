## Net Challenge Project - Dataset 

### GroundingDINO Finetuning Dataset Code 

- This repository contains the code used to create and preprocess the custom dataset for fine-tuning the GroundingDINO model.
- The main code for this project is located in the [gyoenge/net-challenge-groundingdino-finetune](https://github.com/gyoenge/net-challenge-groundingdino-finetune) repository. <br/> Please visit it for more details about the project.

### Description

- hand labeling to annotation.csv :  
    1. prepare folders : 
        - images/
        - annotation/ 
    2. run 
        ```
        python handlabeling_to_anncsv.py
        ```

- yolov8 labeling(txt) to annotation.csv : 
    1. prepare folders : 
        - raw_data/images/ 
        - raw_data/labels/
        - processed_annotation/
    2. run 
        ```
        python yolotxt_to_anncsv.py 
        ```

- custom dataset (video, json) to annotation.csv
    (here we used aihub smoking person dataset)
    1. prepare folders : 
        - raw_data/video/
        - raw_data/label/
        - processed_data/images/
        - processed_data/annotation/
    2. run 
        ```
        python aihub_to_anncsv.py 
        ```

- custom dataset (video, json) to yolov8 lageling(txt) : 
    1. prepare folders : 
        - raw_data/video/
        - raw_data/label/
        - processed_data/images/
        - processed_data/label/
    2. run 
        ```
        python aihub_to_yolo/aihub_to_yolov8txt.py
        ```
