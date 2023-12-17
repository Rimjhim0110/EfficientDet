# EfficientDet: Scalable and Efficient Object Detection

EfficientDet is an efficient and scalable object detection framework that systematically explores neural network architecture design choices to optimize efficiency. This project is an implementation of key components proposed in the research paper, "EfficientDet: Scalable and Efficient Object Detection" by Mingxing Tan, Ruoming Pang, and Quoc V. Le from Google Research. You can access the full paper [here](https://arxiv.org/abs/1911.09070).

![EfficientDet Architecture](Images/EfficientDet_Architecture.jpg)

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Code Organization](#code-organization)
- [References](#references)

## Project Overview

EfficientDet represents a significant evolution in object detection by improving both efficiency and effectiveness. The key features that highlight its significance include:

- **Bi-directional Feature Pyramid Network (BiFPN):** An innovative feature network that allows easy and fast multi-scale feature fusion.
- **Compound Scaling Method:** A novel method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box prediction networks at the same time.
- **State-of-the-Art Performance:** EfficientDet achieves state-of-the-art accuracy on COCO dataset with an order of magnitude fewer parameters and FLOPS.
- **Open-Source Initiative:** The project is open-source, encouraging community contributions and collaboration.
  
## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6+
- PyTorch
- TensorBoard
- OpenCV (cv2)
- pycocotools
  
## Code Organization

The codebase is organized into the following modules:

- `config.py`: Contains configurations and settings for the project, such as COCO classes and colors used for visualization.
- `dataset.py`: Includes data preprocessing, augmentation, as well as loading and handling of the COCO dataset.
- `efficientdet.py`: Contains the core EfficientDet model that defines the architecture and forward pass of the model, including classification and regression heads.
- `loss_function.py`: Contains the implementation of the Focal Loss used for training the object detection model.
- `utils.py`: Contains utility functions used throughout the project, such as BBoxTransform for bounding box transformations, ClipBoxes for clipping boxes to image boundaries, and Anchors for generating anchor boxes.

## References

` https://github.com/google/automl/blob/master/efficientdet 
`
