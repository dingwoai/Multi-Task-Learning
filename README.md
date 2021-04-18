# Multi-Task-Learning
**A list of papers, codes and applications on multi-task learning. Comments and contributions are welcomed!**

And it's updating...

------
## Table of Contents:
- [Papers](#papers) 
  - [Survey](#survey)
  - [Architecture design](#archi)
  - [Loss function](#loss)
  - [Task relationship learning](#trl)
  - [Optimization methods](#optim)
- [Datasets](#datasets)
- [Applications](#apps)
- [Related Areas](#related)

<a name="papers"></a>
## Papers

<a name="survey"></a>
### Survey

- A Survey on Multi-Task Learning. *arXiv*, jul 2017.
- An Overview of Multi-Task Learning in Deep Neural Networks. *arXiv*, jun 2017.
- A brief review on multi-task learning. *Multimedia Tools and Applications*, 77(22):29705–29725, nov 2018.
- Multi-task learning for dense prediction tasks: A survey. *arXiv*, apr 2020.
- A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning. arXiv, jul 2020.
- Multi-task learning with deep neural networks: A survey, 2020.

<a name="archi"></a>
### Architecture design

##### pure hard parameter sharing

- MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving. In *IEEE Intelligent Vehicles Symposium, Proceedings*, 2018.
- Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, pages 7482–7491, 2018.
- UberNet: Training a universal convolutional neural network for Low-, Mid-, and high-level vision using diverse datasets and limited memory. *Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017*, 2017-January:5454–5463, 2017.
- Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD ’18, page 1930–1939, New York, NY, USA, 2018. Association for Computing Machinery.

##### pure soft parameter sharing

- Cross-Stitch Networks for Multi-task Learning. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2016.
- Deep Multi-task Representation Learning: A Tensor Factorisation Approach. *5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings*, may 2016. [Code](https://github.com/wOOL/DMTRL)
- Latent multi-task architecture learning. In *33rd AAAI Conference on Artificial Intelligence, AAAI 2019, 31st Innovative Applications of Artificial Intelligence Conference, IAAI 2019 and the 9th AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019*, 2019. [Code](https://github.com/sebastianruder/sluice-networks)
- NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3200–3209. IEEE, jun 2019. [Code](https://github.com/ethanygao/NDDR-CNN)

##### a mix of hard and soft

- End-to-end multi-task learning with attention. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2019-June:1871–1880, 2019. [Code](https://github.com/lorenmt/mtan)
- Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 675–684, 2018.
- Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4101–4110, 2019.
- Mti-net: Multi-scale task interaction networks for multi-task learning. In *ECCV*, 2020. [Code](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)
- Attentive Single-Tasking of Multiple Tasks. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1851–1860. IEEE, jun 2019. [Code](https://github.com/facebookresearch/astmt)
- Adashare: Learning what to share for efficient deep multi-task learning. *ArXiv*, abs/1911.12423, 2020.
- Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. [Code](https://github.com/bhpfelix/MTLNAS)
- Many Task Learning With Task Routing. In*2019IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1375–1384. IEEE, oct 2019.

<a name="loss"></a>
### Loss function

- Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, pages 7482–7491, 2018.
- Auxiliary Tasks in Multi-task Learning. *arXiv*, may 2018.
- GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. *35th International Conference on Machine Learning, ICML 2018*, 2:1240–1251, 2018.
- Self-paced multi-task learning. *AAAI Conference on Artificial Intelligence*, pages 2175–2181, 2017.
- Dynamic task prioritization for multitask learning. ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018.
- Focal Loss for Dense Object Detection. In *Proceedings of the IEEE International Conference on Computer Vision*, 2017.
- End-to-end multi-task learning with attention. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2019-June:1871–1880, 2019. [Code](https://github.com/lorenmt/mtan)
- MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, volume 2019-June, pages 1200–1210. IEEE, jun 2019.
- Dynamic Task Weighting Methods for Multi-task Networks in Autonomous Driving Systems. In *2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)*, pages 1–8. IEEE, sep 2020.
- A Comparison of Loss Weighting Strategies for Multi task Learning in Deep Neural Networks. *IEEE Access*, 7:141627–141632, 2019.

<a name="trl"></a>
### Task relationship learning

<a name="optim"></a>
### Optimization Methods

Also I need to mention that many MTL approaches utilize not just one category of methods listed above but a combination instead. 

<a name="datasets"></a>
## Datasets

Commonly used in computer vision:

-  [Taskonomy](http://taskonomy.vision/) is currently the largest dataset specifically designed for multi-task learning. It has about 4.5 million images of indoor scenes from 3D scans of about 600 buildings and every image has an annotation for all 26 tasks.
-  [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) is a large-scale dataset for indoor scenes understanding, which contains a variety of computer vision tasks. There are in total 1449 densely labeled RGBD images, capturing 464 diverse indoor scenes, with 35,064 distinct objects from 894 different classes.
- MultiMNIST is an MTL version of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. It is formed by overlaying multiple handwrit- ten digit images together. One of these is placed at the top-left while the other at the bottom-right. The tasks are classifying simultaneously the digit on the top-left and on the bottom-right.
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 10,000 identities, each has 20 images, which result a total number of 200,000 images. Since CelebA is annotated with 40 face attributes and 5 key points, it can be used in MTL setting by considering each attribute as a distinct classification task.
- [Cityscapes](https://www.cityscapes-dataset.com/) dataset is established for semantic urban scene understanding. It is comprised of 5000 images with high quality pixel-level annotations as well as 20,000 additional images with coarse annotations. Tasks like semantic segmentation, instance segmentation and depth estimation are able to be trained together on Cityscapes.
- [MS-COCO](http://mscoco.org/) is a widely used dataset in CV. It contains 382k images with a total of 2.5 million labeled instances spanning 91 objects types. It can be used for multiple tasks including image classification, detection and segmentation.
- [KITTI](www.cvlibs.net/datasets/kitti) is by far the most famous and commonly used dataset for autonomous driving. It provides benchmarks for multiple driving tasks: e.g. stereo matching, optical flow estimation, visual odometry/SLAM, semantic segmentation, object detection/orientation estimation and object tracking.
- [BDD100K](https://bdd-data.berkeley.edu/) is a recent driving dataset designed for heterogeneous multitask learning. It is comprised of 100K video clips and 10 tasks: image tagging, lane detection, drivable area segmentation, road object detection, semantic segmentation, instance segmentation, multi-object detection tracking, multi-object segmentation tracking, domain adaptation and imitation learning.

<a name="apps"></a>
## Applications

#### Natural language processing

#### Speech processing

#### Computer vision

#### Reinforcement learning

#### Recommendation

#### Multi-modal

- Towards General Purpose Vision Systems, https://arxiv.org/pdf/2104.00743.pdf


<a name="related"></a>
## Related Areas

- Transfer Learning
- Auxiliary Learning
- Multi-label Learning
- Multi-modal Learning
- Meta Learning
- Continual Learning
- Curriculum Learning
- Ensemble, Distillation and Model Fusion