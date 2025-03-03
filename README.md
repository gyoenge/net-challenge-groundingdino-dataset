## groundingDINO finetune dataset

- 2023 net challenge AI dataset code
- for groundingDINO finetune : https://github.com/gyoenge/net-challenge-groundingdino-finetune

---

### description

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

