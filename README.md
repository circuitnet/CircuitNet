# CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)

## Overview 
This repository is intended to hosts codes and demos for CircuitNet, we hope this codebase would be helpful for users to reproduce exiting methods. Pretrianed weights will be available soon.
<p align="center">
  <img src="assets/overall_structure.png" height=300>
</p>

```
usage: test.py [--task TASK_NAME]  [--pretrained PRETRAINED_WEIGHTS_PATH] 
```
Change the configure to fit your file path and hyper-parameter in [config.py](utils/configs.py).

More information about the dataset can be accessed from our [web page](https://circuitnet.github.io/).

## Prerequisites

All dependencies can be installed using pip:

```sh
python -m pip install -r requirements.txt
```

Our experiments run on Python 3.9 and PyTorch 1.11. Other versions should work but are not tested.

## Data Preparation
Please follow the instructions in [the quick start page](https://circuitnet.github.io/intro/quickstart.html) to setup the CircuitNet dataset for specific task(Congestion/DRC/IR Drop).

## Example Usage:

### Test

#### Congestion
```python
python test.py --task congestion_gpdl --pretrained PRETRAINED_WEIGHTS_PATH
```


#### DRC
```python
python test.py --task drc_routenet --pretrained PRETRAINED_WEIGHTS_PATH --save_as_npy
```

#### IR Drop
```python
python test.py --task irdrop_mavi --pretrained PRETRAINED_WEIGHTS_PATH --save_as_npy
```


### Train
#### Congestion
```python
python train.py --task congestion_gpdl --save_path work_dir/congestion_gpdl/
```


#### DRC
```python
python train.py --task drc_routenet --save_path work_dir/drc_routenet/
```

#### IR Drop
```python
python train.py --task irdrop_mavi --save_path work_dir/irdrop_mavi/
```

## License
This repository is released under the BSD 3-Clause. license as found in the LICENSE file.


## Citation
If you think our work is useful, please feel free to cite [our paper](https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3571-8) 😆 :
```
@article{chai2022circuitnet,
  title = {CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)},
  author = {Chai, Zhuomin and Zhao, Yuxiang and Lin, Yibo and Liu, Wei and Wang, Runsheng and Huang, Ru},
  journal= {SCIENCE CHINA Information Sciences},
  year = {2022}
}
```

## Contact
For any question, please feel free to contact us.

```
Zhuomin Chai: zhuominchai@whu.edu.cn
Yuxiang Zhao: yuxiangzhao@stu.pku.edu.cn
```
