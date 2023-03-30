# CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)

## Overview

This repository is intended to hosts codes and demos for CircuitNet, we hope this codebase would be helpful for users to reproduce exiting methods. More information about the dataset can be accessed from our web page [https://circuitnet.github.io/](https://circuitnet.github.io/).

<p align="center">
  <img src="assets/overall_structure.png" height=300>
</p>

--------

## ChangeLog
- 2022/8/1 
  
  First release.
- 2022/9/6 

  Pretrained weights are available in [Google Drive](https://drive.google.com/drive/folders/10PD4zNa9fiVeBDQ0-drBwZ3TDEjQ3gmf?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1dUEt35PQssS7_V4fRHwWTQ?pwd=7i67).
- 2022/12/12 

  Graph features are available in the graph_features dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/s/1udXVZnfjqniH9paKfyc2eQ?pwd=ijdh).
- 2022/12/29 

  LEF/DEF (sanitized) are available in the LEF&DEF dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/s/1udXVZnfjqniH9paKfyc2eQ?pwd=ijdh).

- 2023/3/22 

  LEF/DEF is updated to include tech information (sanitized). Each tarfile contains 500 DEF files and can be decompressed separately. We also provide example DEF files.
  
  Congestion features and graph features generated from ISPD2015 benchmark are available in the ISPD2015 dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/s/1udXVZnfjqniH9paKfyc2eQ?pwd=ijdh).

## Prerequisites

Dependencies can be installed using pip:

```sh
pip install -r requirements.txt
```

PyTorch is not included in requirement.txt, and you could intall it follow the instruction in PyTorch homepage [https://pytorch.org/](https://pytorch.org/).

Our experiments run on Python 3.9 and PyTorch 1.11. Other versions should work but are not tested.

## Data Preparation

Please follow the instructions in the [quick start page](https://circuitnet.github.io/intro/quickstart.html) to setup the CircuitNet dataset for specific task(Congestion/DRC/IR Drop).

## Example Usage:

**Change the configure in [utils/config.py](utils/configs.py) to fit your file path and adjust hyper-parameter before starting.**

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

If you think our work is useful, please feel free to cite our [paper](https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3571-8)😆 .

```
@article{chai2022circuitnet,
    title = {CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)},
    author = {Chai, Zhuomin and Zhao, Yuxiang and Lin, Yibo and Liu, Wei and Wang, Runsheng and Huang, Ru},
    journal= {SCIENCE CHINA Information Sciences},
    volume={65},
    number = "12",
    pages={227401-},
    year = {2022}
}
```

## Contact

For any questions, please do not hesitate to contact us.

```
Zhuomin Chai: zhuominchai@whu.edu.cn
Yuxiang Zhao: yuxiangzhao@stu.pku.edu.cn
Yibo Lin: yibolin@pku.edu.cn
```
