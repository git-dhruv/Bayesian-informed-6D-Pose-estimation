# Bayesian Informed 6D Pose estimation
We introduce a novel framework for addressing the challenges in 6D pose estimation for pick-and-place manipulation. The proposed vision-based approach leverages a Bayesian combination of SE(3) tracking and SO(3) estimation. This aims to achieve a balance between accuracy and computation time, a common trade-off in existing solutions. Experimental results demonstrate superior performance compared to current state-of-the-art methods, highlighting the proposed framework's effectiveness in high-frequency pose estimation. This is the code base for the project. 


## Setup
The following setup works on Linux only. However, you can modify the CMakeLists.txt yourself to get it running on your OS.

### Classical 6D Pose Setup

#### Dependency requirements
Please install the following in your system

- pcl>=1.3
- eigen>=3.3.9
- openCV
- Boost

#### Building the code
```
cd classical_6d
mkdir build
cmake ..
make -j8 
```

### Python code setup
We have provided a requriements file for setting up the python environment. 
```
mkdir venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```


### Data to download
We are using mustard data. Download the following:
YCB Video Data: https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_Video_data_organized/0050.tar.gz
Ply Files: https://archive.cs.rutgers.edu/pracsys/se3_tracknet/YCB_models_with_ply.zip
Synthetic Training Data: https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_traindata/mustard_bottle.tar.gz
Weights: Download our weights from Supplmentary Materials.

You are ready to run the code now! Please ensure that the paths are setup correctly for reading the inference datasets. 
## Running 

```
python3 src\main.py
```


## Code Structure
|-- classical_6d
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 6D Pose estimation (Classical) Pipeline
|
|-- configurations
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Configuration Files
|
|-- logs
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Log Files generated from inference. Don't confuse these with lightning logs!
|
|-- models
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Neural Networks that are used in inference
|
|-- notebooks
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Contains data preprocessing, depth simulator, and result generation script
|
|-- results
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Stored various plots for report
|
|-- src
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Utils
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Dataloader
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Inference
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Kalman Filter
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Renderer
|
|-- training
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Training pipeline
|
|-- weights
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- Saved Weights


## Bugs
As of now we don't have a bug reporting mechanism setup. Please feel free to email the authors, we are more than happy to help you. Please be aware that we do not guarentee any support for MacOS due to our limited knowledge. 

## Authors
Dhruv Parikh

## Acknowledgements
Special Thanks to the Dr. Wen for open sourcing their code on OpenGL rendering! 

## License 
Copyright <2023> <University of Pennsylvania>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


