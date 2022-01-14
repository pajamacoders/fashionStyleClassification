# FashionStyleClassification model based on ASEN++ 
This repository include pytorch implementation of fashionStyleClassification model.
The base model used is [ASEN++](https://github.com/Maryeon/asenpp).
Simple modification applied to resnet backbone and child class is added to train it as a classification model.
The attribute specific feature is combined with global average feature to improve style classification performance.


## Environments

- **Ubuntu** Linux 36eee03bfc57 3.10.0-1160.41.1.el7.x86_64
- **CUDA**  Build cuda_11.2.r11.2/compiler.29373293_0
- **Python** 3.8

Install other required packages by

```sh
pip install -r requirements.txt
```

To install mmcv. mmcv version used in this repo is 'mmcv==1.3.18'
```sh
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e .
```


## Datasets
The dataset used in this repo is [KFashion](https://aihub.or.kr/aidata/7988) dataset.
I made a small modification to the style class to simplify the classes from 14 to 3.
Formal, semi-formal, casual.

### Layout

After unzip the `data.zip`, a directory rooted as `data` is created and has a layout as following:

```sh
data
├── Kfashion
│   ├── candidate_test.txt
│   ├── candidate_valid.txt
│   ├── filenames_train.txt
│   ├── filenames_valid.txt
│   ├── label_train.txt
│   └── query_valid.txt
└── meta.json
```

### Download Data


### Configuration

The behavior of our codes is controlled by configuration files under the `config` directory. 

```sh
config
└── KFashion
    ├── KFashion.yaml
    ├── s1.yaml
    └── s2.yaml
```

Each dataset is configured by two types of configuration files. One is `<Dataset>.yaml` that specifies basic dataset information such as path to the training data and annotation files. The other two set some training options as needed.

If the above `data` directory is placed at the same level with `main.py`, no changes are needed to the configuration files. Otherwise, be sure to correctly configure relevant path to the data according to your working environment.

## Training

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s1.yaml
```

Based on the trained global branch, the second stages jointly train the whole ASEN:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s1/checkpoint.pth.tar
```

## Evaluation

Run the following script to test on the trained models:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s2/model_best.pth.tar --test TEST
```





