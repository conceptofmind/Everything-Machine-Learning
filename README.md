# Everything-Machine-Learning

## Libraries / Repositories

### Computer Vision

#### [YOLOv5](https://github.com/ultralytics/yolov5)

### Natural Language Processing

#### [Natural Language Toolkit](https://www.nltk.org/index.html#)

### Audio

#### [wav2vec 2.0](https://github.com/facebookresearch/fairseq)
- [Paper](https://arxiv.org/abs/2006.11477)
- [SPEECH RECOGNITION WITH WAV2VEC2](https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html)

#### [Tacotron 2 (without wavenet)](https://github.com/NVIDIA/tacotron2)
- [Paper](https://arxiv.org/pdf/1712.05884.pdf)
- [TEXT-TO-SPEECH WITH TACOTRON2](https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html#sphx-glr-tutorials-tacotron2-pipeline-tutorial-py)

#### [CycleGAN-VC3-PyTorch](https://github.com/jackaduma/CycleGAN-VC3)
- [Paper](https://arxiv.org/abs/2010.11672)
- [Project Page](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html)

### Generative

#### [stylegan3](https://github.com/NVlabs/stylegan3)
- [Paper](https://arxiv.org/abs/2106.12423)
- [Project Page](https://nvlabs.github.io/stylegan3/)

## Books

### Natural Language Processing

#### [Natural Language Processing with Python](https://www.nltk.org/book/)

### TensorFlow

####


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

## Research Papers

### Image Generation

#### [DALL-E](https://arxiv.org/abs/2102.12092)


## Activation Functions

### GLU
```
GLU(a,b)=a⊗σ(b)
```
#### [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html)
```python
m = torch.nn.GLU(dim=- 1)
input = torch.randn(4, 2)
output = m(input)
```
