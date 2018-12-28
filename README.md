# 評分環境
- 機器: NCHC Aitrain container
- Python environment as LAB5
# 使用方法
- Download this repository to benchmark your project
```shell
$ git clone https://github.com/nctu-arch/NCTU_DLSR_final_project.git
```
- Preparation: install requirements
```shell
$ cd NCTU_DLSR_final_project
$ pip3 install -r requirements.txt
```
- import benchmark
```python
from benchmark import benchmarking
```
- Benchmarking function usage - `benchmarking(team, task, model, preprocess_fn, *pre_args, **pre_kwargs)`
  - team: 
      - 1~12
  - task: 
    - 0: classification
    - 1: super resolution
    - 2: objection detection
  - model: pytorch 
    - 目前計算 pytorch model weight 數量及大小
  - preprocess_fn, *pre_args, **pre_kwargs: 
    - 前處理 function， 可以轉換 data format
    - 參數 可自定義, 無則 None
- 撰寫 Inference code
```python
net = resnet18() # define model 
@benchmarking(team=12, task=0, model=net, preprocess_fn=None)
def inference_fn(*args, **kwargs):
    dev = kwargs['device']
    if dev == 'cpu':
        metric = do_cpu_inference()
        ...
    elif dev == 'cuda':
        metric = do_gpu_inference()
        ...
    return metric
```
# Test Categories
* CINIC-10
    * Baseline
        * CINIC-10 test data
    * Accuracy Ranking
        * private test data
    * Model size
    * CPU inference time
    * GPU inference time
* DIV2K
    * Baseline
        * DIV2K x2 validtion data
    * PSNR Ranking
        * private test data
    * Model size
    * CPU inference time
    * GPU inference time
* Clothes
    * Baseline: 
        * Validation data
    * F-score Ranking
        * ITRI test data
    * Model size
    * CPU inference time
    * GPU inference time
# What to Submit?
* Any source code you used in your project.
* Create a team directory named teamX including 'Classification','Object Detection' and 'Super Resolution' to push each task respectively.
```
e.g..
.
├── team11
└── team12
    ├── Classification
    ├── Object Detection
    └── Super Resolution
```
# How to Submit?
* As a student, you can apply for a GitHub Student Developer Pack, which offers unlimited private repositories.
* Fork this repository, and then make your forked repo private. (Settings -> Danger Zone)
* Add nctu-arch as collaborator. (Settings -> collaborator)
* After deadline you need to create a pull request for open review.
* Please describe the external plugins you used and its usage precisely.
# Example usage
  ```
  TESTDATADIR="./path/to/cifar10/data" python3 example.py
  ```
  * default: overide team 12 data
# Score sheet link
https://docs.google.com/spreadsheets/d/1-Z8Q7NkgjgPS7OUu3Z0AQiBScHSKcc-IoEZdGazyq1A/edit?usp=sharing
