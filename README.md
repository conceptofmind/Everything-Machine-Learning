# Everything-Machine-Learning
This is just a compilation of resources and references I found useful for Machine Learning. Not a Roadmap or advice. I just needed to clean up my 100 browser tabs.

## Libraries / Repositories

### Geometry

#### PyTorch Geometric
- [Documentation](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

### Natural Language Processing

#### [Natural Language Toolkit](https://www.nltk.org/index.html#)

### Computer Vision

#### [YOLOv5](https://github.com/ultralytics/yolov5)

### Audio

#### [wav2vec 2.0](https://github.com/facebookresearch/fairseq)
- [Paper](https://arxiv.org/abs/2006.11477)
- PyTorch Tutorial - [SPEECH RECOGNITION WITH WAV2VEC2](https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html)

#### [Tacotron 2 (without wavenet)](https://github.com/NVIDIA/tacotron2)
- [Paper](https://arxiv.org/pdf/1712.05884.pdf)
- PyTorch Tutorial - [TEXT-TO-SPEECH WITH TACOTRON2](https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html#sphx-glr-tutorials-tacotron2-pipeline-tutorial-py)

#### [CycleGAN-VC3-PyTorch](https://github.com/jackaduma/CycleGAN-VC3)
- [Paper](https://arxiv.org/abs/2010.11672)
- [Project Page](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html)

### Generative

#### [stylegan3](https://github.com/NVlabs/stylegan3)
- [Paper](https://arxiv.org/abs/2106.12423)
- [Project Page](https://nvlabs.github.io/stylegan3/)

### Robotics

#### [Underactuated Robotics](http://underactuated.csail.mit.edu/index.html)
- [Course Page](http://underactuated.csail.mit.edu/Spring2021/resources.html#further_material)
- [YouTube Channel](https://www.youtube.com/channel/UChfUOAhz7ynELF-s_1LPpWg)

## Books

### Natural Language Processing

#### [Natural Language Processing with Python](https://www.nltk.org/book/)

### TensorFlow

####

### Reinforcement Learning

#### [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf)

### Robotics

#### [A Mathematical Introduction to Robotic Manipulation](https://atc.home.ece.ust.hk/files/mls94-complete.pdf)


## Datasets

### Image

#### [Multimedia Commons](http://www.multimediacommons.org/) 
- [Paper](https://arxiv.org/abs/1503.01817)
- [Installer](https://pypi.org/project/yfcc100m/)

#### [COCO](https://cocodataset.org/#download)
- [Paper](https://arxiv.org/abs/1405.0312)
- [pycocoDemo](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb)

### Audio

#### [speech.ko](https://github.com/homink/speech.ko)

#### [KoSpeech](https://github.com/sooftware/kospeech)
- [Paper](https://www.sciencedirect.com/science/article/pii/S2665963821000026)

#### [TEDxJP-10K ASR Evaluation Dataset](https://github.com/laboroai/TEDxJP-10K)


## Accelerated / High Performance Computing

#### [Numba](https://numba.pydata.org/)
- [Documentation](https://numba.readthedocs.io/en/stable/index.html#)
- [Github Page](https://github.com/numba/numba/)
- [Numba: High-Performance Python with CUDA Acceleration](https://developer.nvidia.com/blog/numba-python-cuda-acceleration/)
- [Seven Things You Might Not Know about Numba](https://developer.nvidia.com/blog/seven-things-numba/)
- [GPU-Accelerated Graph Analytics in Python with Numba](https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/)
- [gtc2017-numba](https://github.com/ContinuumIO/gtc2017-numba)

#### [Rapids](https://rapids.ai/)
- [Documentation](https://docs.rapids.ai/)
- [Github Page](https://github.com/rapidsai)

#### [CUDA Python](https://developer.nvidia.com/cuda-python#:~:text=CUDA%20Python%20provides%20uniform%20APIs%20and%20bindings%20for,from%20Preferred%20Networks%2C%20for%20GPU-accelerated%20computing%20with%20Python.)
- [Documentation](https://nvidia.github.io/cuda-python/overview.html)

#### [Horovod](https://github.com/horovod/horovod#install)
- [Documentation](https://horovod.readthedocs.io/en/stable/)

#### [Fairscale](https://github.com/facebookresearch/fairscale)
- [Documentation](https://fairscale.readthedocs.io/en/latest/)

#### [DeepSpeed](https://www.deepspeed.ai/)


## Research Papers

### Image Generation

#### [DALL-E](https://arxiv.org/abs/2102.12092)

## Mathematics for Machine Learning

### [First Order Differential equations](https://www.khanacademy.org/math/differential-equations/first-order-differential-equations)

## College Courses

### Artficial Intelligence
- [UC Berkeley CS188 Intro to AI](http://ai.berkeley.edu/reinforcement.html#Introduction)

### Reinforcement Learning
- [UCL Course on RL](https://www.davidsilver.uk/teaching/)
- [CS 285 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
- 

## Activation Functions

### GLU
```
GLU(a,b)=a?????(b)
```
#### [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html)
```python
m = torch.nn.GLU(dim=- 1)
input = torch.randn(4, 2)
output = m(input)
```

## APIs

### [OpenAI](https://openai.com/api/)

### [Alpaca](https://alpaca.markets/)
- [Paper Trading](https://alpaca.markets/docs/trading/paper-trading/)
- [OAuth](https://alpaca.markets/docs/oauth/)
- [Trade Stocks in Your Browser Using Google Colab and Alpaca API](https://alpaca.markets/learn/trade-with-google-colab/)
- [Integrating Alpaca into an Android App with AppAuth Part 1](https://alpaca.markets/learn/android-dashboard-01/)
- [Connect and Trade Smart](https://alpaca.markets/learn/marketplug-websockets/)
- [How to Use OAuth Support?](https://alpaca.markets/learn/oauth-guide/)
- [Alpaca Trading API Guide ??? A Step-by-step Guide](https://algotrading101.com/learn/alpaca-trading-api-guide/)

### Quant Connect
- [Live Trading on QuantConnect Now Free for Students](https://www.quantconnect.com/blog/quantconnect-live-trading-now-free-for-students/)
- [https://www.quantconnect.com/docs/key-concepts/developing-in-the-ide](https://www.quantconnect.com/docs/key-concepts/developing-in-the-ide)
- [Introduction to Financial Python](https://www.quantconnect.com/tutorials/tutorial-series/introduction-to-financial-python)

## Blogs

### Finance

#### Coding Finance
- [Time Value of Money in Python](https://www.codingfinance.com/post/2018-03-19-tvm_py/)
