# Sample code for ICISC 2023 phlexing.

## Overview

This branch hosts the sample codes for [ICISC 2023 Phlexing "Machine Learning driven Static IR drop estimation of SoC power grid network"](https://edaoss.icisc.cn/file/cacheFile/ae0e578a501d47b6b899bf5a1bfe20d9.pdf).
The dataset can be downloaded at [Baidu Netdisk](https://pan.baidu.com/s/1B9Hl2PHJ5f5KW-vpms3Vfw?pwd=75tx).

Use git checkout to switch to this branch.

```sh
git checkout icisc_2023
```


## Prerequisites

Dependencies can be installed using pip:

```sh
pip install -r requirements.txt
```

PyTorch is not included in requirement.txt, and you could install it following the instruction on PyTorch homepage [https://pytorch.org/](https://pytorch.org/).

Our experiments run on Python 3.9 and PyTorch 1.11. Also works on the environment on test server, which has Python 3.11 and PyTorch 2.0.

## IR Drop Prediction

Here is the flow chart of this sample code. This is just one implementation, you are encouraged to explore your own implementation.

<p align="center">
  <img src="flow.png" height=300>
</p>

### Data Preparation

There are 2 sample data in /feature_extraction/data, which are the reports from GloryBolt, containing power, resistance and IR drop. We will use them for preprocessing and training.

Please run the following codes one by one under directory /feature_extraction.

Parsing the report and turn them into images (numpy array) through tiling.

```python
python process_data.py
```

Preprocessing the numpy array to facilitate model training.

```python
python generate_training_set.py
```

Generating the annotation csv to control the data allocation.

```python
python generate_csv.py
```

### Model Training and Testing.

We use a simple FCN as example, which takes in effective resistance and power, and output IR drop prediction.

Model Training (under directory /sample_model_training). Models and logs are saved in /work_dir (auto generated after training) by default.

```python
python train.py --args args/train.json
```

Model Testing. **Remember to modify the "--pretrained" argument to point to the model saved in /work_dir.**

```python
python test.py --args args/test.json --pretrained PATH_TO_SAVED_MODEL
```

Test results, including visualizations and report, are saved in /work_dir by default.


**The default setting use GPU, and if you don't have one, add "--cpu" argument.**

For example:
```python
python train.py --args args/train.json --cpu
```

## Change Log.
- 2023/11/6

  add final_test argument in /feature_extraction/process_data.py and /sample_model_training/test.py.

  final_test=False:
  The origin test flow, use the instance name in static_ir report to ensure the length of predicted report match the golden report, then the metric can be calculated directly.
        
  final_test=True:
  staitc_ir rpt is not provided in the final test, and the predicted report's length does not have to match the golden report. Then the instance name in the power.rpt is used to output the report. A dedicated script will be used to calculate metric.

- 2023/11/3

  Update the format of the output file.

- 2023/9/22

  Update to include prediction for GND_bounce. 

- 2023/9/20

  Update to ensure compatibility with Windows and CPU.

- 2023/9/13

  First release.
  

