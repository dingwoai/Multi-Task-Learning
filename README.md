# Multi-Task-Deep-Learning
**A list of papers, codes and applications on multi-task deep learning. Comments and contributions are welcomed!**

And it's updating...

------
## Table of Contents:
- [Papers](#papers) 
  - [Survey](#survey)
  - [Theory](#theory)
  - [Architecture design](#archi)
    - [Pure hard](#hard)
    - [Pure soft](#soft)
    - [Mixture](#mix)
    - [Architecture search](#nas)
    - [Dynamic architecture](#dynamic)
    - [Probabilistic MTL](#proba)
  - [Task relationship learning](#trl)
  - [Optimization methods](#optim)
    - [Loss function](#loss)
    - [Optimization](#optimization)
  - [Novel Settings](#novel)
- [Datasets](#datasets)
- [Applications](#apps)
- [Related Areas](#related)
- [Trends](#trends)

<a name="papers"></a>

## Papers

<a name="survey"></a>
### Survey

- [1997] Caruana, R. Multitask Learning. *Machine Learning* **28,** 41–75 (1997). https://doi.org/10.1023/A:1007379606734.
- http://www.siam.org/meetings/sdm12/zhou_chen_ye.pdf Multi-Task Learning: Theory, Algorithms, and Applications (2012, SDM tutorial)
- A Survey on Multi-Task Learning. *arXiv*, jul 2017.
- An Overview of Multi-Task Learning in Deep Neural Networks. *arXiv*, jun 2017.
- A brief review on multi-task learning. *Multimedia Tools and Applications*, 77(22):29705–29725, nov 2018.
- Multi-task learning for dense prediction tasks: A survey. *arXiv*, apr 2020.
- A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning. arXiv, jul 2020.
- Multi-task learning with deep neural networks: A survey, 2020.
- [arXiv 2022] Multi-Task Learning for Visual Scene Understanding. [paper](https://arxiv.org/abs/2203.14896)
  - PhD Thesis.
- [arXiv 2022] A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods. [paper](https://arxiv.org/abs/2204.03508)

<a name="theory"></a>

### Theory

- [2019] How to study the neural mechanisms of multiple tasks, [paper](https://www.sciencedirect.com/science/article/pii/S2352154619300695)
- [ICLR 2021] Deciphering and Optimizing Multi-Task Learning: a Random Matrix Approach, https://openreview.net/forum?id=Cri3xz59ga
- [ICML 2021] Bridging Multi-Task Learning and Meta-Learning: Towards Efficient Training and Effective Adaptation, [paper](https://arxiv.org/abs/2106.09017), [code](https://github.com/AI-secure/multi-task-learning)
- [bioRxiv 2021] Abstract representations emerge naturally in neural networks trained to perform multiple tasks, [paper](https://www.biorxiv.org/content/10.1101/2021.10.20.465187v1)

<a name="archi"></a>

### Architecture design

<a name="hard"></a>

#### pure hard parameter sharing

- [ICCV 2017] Multi-task Self-Supervised Visual Learning. [paper](https://ieeexplore.ieee.org/document/8237488)
- MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving. In *IEEE Intelligent Vehicles Symposium, Proceedings*, 2018.
- Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, pages 7482–7491, 2018.
- UberNet: Training a universal convolutional neural network for Low-, Mid-, and high-level vision using diverse datasets and limited memory. *Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017*, 2017-January:5454–5463, 2017.
- Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD ’18, page 1930–1939, New York, NY, USA, 2018. Association for Computing Machinery.
- [CVPR 2018] PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning, [Paper](https://arxiv.org/abs/1711.05769), [Code](https://github.com/arunmallya/packnet)
- [ECCV 2018] Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights, [Paper](https://arxiv.org/abs/1801.06519v2), [Code](https://github.com/arunmallya/piggyback)
  - learn to mask weights of an existing network
- [ICRA 2019] Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations. [paper](https://arxiv.org/abs/1809.04766)
  - Using aysmmetric datasets with uneven numbers of annotations with knowledge distillation (under the assumption of a powerful teacher network).
- [CCN 2019] Modulation of early visual processing alleviates capacity limits in solving multiple tasks, [Paper](https://arxiv.org/abs/1907.12309)
  - By associating neural modulations with task-based switching of the state of the network and characterizing when such switching is helpful in early processing, our results provide a functional perspective towards understanding why task-based modulation of early neural processes might be observed in the primate visual cortex.
- [CVPR 2019] Attentive Single-Tasking of Multiple Tasks, [paper](https://arxiv.org/abs/1904.08918), [code](http://vision.ee.ethz.ch/~kmaninis/astmt/)
  - We refine features with a task-specific residual adapter branch (RA) and attend to particular channels with task-specific Squeeze-and-Excitation (SE) modulation.
  - We also **enforce the task gradients to be statistically indistinguishable through adversarial training**.
- [AAAI 2020] Learning Sparse Sharing Architectures for Multiple Tasks, [paper](https://arxiv.org/abs/1911.05034)
- [ICML 2020] Learning to Branch for Multi-Task Learning, http://proceedings.mlr.press/v119/guo20e.html.
- [arXiv 2021] UniT: Multimodal Multitask Learning with a Unified Transformer, https://arxiv.org/abs/2102.10772, [Code](https://mmf.sh/)
- [arXiv 2021] You Only Learn One Representation: Unified Network for Multiple Tasks, [paper](https://arxiv.org/abs/2105.04206), [Code](https://github.com/WongKinYiu/yolor)
- [arXiv 2021] Spatio-Temporal Multi-Task Learning Transformer for Joint Moving Object Detection and Segmentation. [paper](https://arxiv.org/abs/2106.11401)
- [NeurIPS 2021] MTL-TransMODS: Cascaded Multi-Task Learning for Moving Object Detection and Segmentation with Unified Transformers. [paper](https://ml4ad.github.io/files/papers2021/MTL-TransMODS:%20Cascaded%20Multi-Task%20Learning%20for%20Moving%20Object%20Detection%20and%20Segmentation%20with%20Unified%20Transformers.pdf)
- [NeurIPS 2021] SOLQ: Segmenting Objects by Learning Queries. [paper](https://arxiv.org/abs/2106.02351), [code](https://github.com/megvii-research/SOLQ)
  - Based on but outperformed DETR (on detection) with **learned unified queries** for instance class, location and mask.
  - Mask branch is supervised with DCT-compressed representation.
- [CVPR 2021] CompositeTasking: Understanding Images by Spatial Composition of Tasks, [paper](https://arxiv.org/abs/2012.09030), [Code](https://github.com/nikola3794/composite-tasking): One network for multiple tasks, but requires multiple inferences.
- [ICCV 2021] Multi-Task Self-Training for Learning General Representations, [paper](https://arxiv.org/abs/2108.11353)
  - Multi-task self-training with pseudo labels (generated by multiple single-task teachers)
  - Cross training on multiple vision datasets
- [ICCV 2021] MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning using an Anchor Free Approach, [paper](https://arxiv.org/abs/2108.05060)
- [arXiv 2021] Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning in Dynamic Environments. [paper](https://arxiv.org/abs/2201.00042)
  - Sparse representation.
- [ECCV 2022] Inverted Pyramid Multi-task Transformer for Dense Scene Understanding. [paper](https://arxiv.org/abs/2203.07997). [code](https://github.com/prismformore/InvPT)
- [arXiv 2022] Multitask Emotion Recognition Model with Knowledge Distillation and Task Discriminator. [paper](https://arxiv.org/abs/2203.13072)
  - Multi-task model with gradient reversal layer and task disciminator.
- [arXiv 2022] M^2BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation. [paper](https://arxiv.org/abs/2204.05088). [project](https://nvlabs.github.io/M2BEV/)
  - Comparison with LSS: 2D to 3D transformation without estimating depth. (Each pixel in 2D feature map is mapped to a set of points in the camera ray in 3D space).
  - Multi-tasking 3D detection and BEV segmentation causes slight drop in performance.
- [ICME 2022] Rethinking Hard-Parameter Sharing in Multi-Domain Learning. [paper](https://arxiv.org/abs/2107.11359)
- [NeurIPS 2022] Effective Adaptation in Multi-Task Co-Training for Unified Autonomous Driving. [paper](https://arxiv.org/abs/2209.08953)
  - A LV-Adapter incorporates language priors in the multi-task model via task-specific prompting and alignment between visual and textual features.
- [arXiv 2023] A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision. [paper](https://arxiv.org/abs/2303.17376)

<a name="soft"></a>

#### pure soft parameter sharing

- Cross-Stitch Networks for Multi-task Learning. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2016.
- Deep Multi-task Representation Learning: A Tensor Factorisation Approach. *5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings*, may 2016. [Code](https://github.com/wOOL/DMTRL)
- [AAAI 2019] Latent multi-task architecture learning. [paper](https://ojs.aaai.org//index.php/AAAI/article/view/4410), [Code](https://github.com/sebastianruder/sluice-networks)
- NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3200–3209. IEEE, jun 2019. [Code](https://github.com/ethanygao/NDDR-CNN)

<a name="mix"></a>

#### a mix of hard and soft

- End-to-end multi-task learning with attention. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2019-June:1871–1880, 2019. [Code](https://github.com/lorenmt/mtan)
- Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 675–684, 2018.
- Pattern-affinitive propagation across depth, surface normal and semantic segmentation. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4101–4110, 2019.
- Mti-net: Multi-scale task interaction networks for multi-task learning. In *ECCV*, 2020. [Code](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)
- Attentive Single-Tasking of Multiple Tasks. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1851–1860. IEEE, jun 2019. [Code](https://github.com/facebookresearch/astmt)
- Many Task Learning With Task Routing. In*2019IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1375–1384. IEEE, oct 2019.
- [CVPR 2022] Task Adaptive Parameter Sharing for Multi-Task Learning. [paper](https://arxiv.org/abs/2203.16708)
  - Differentiable task-specific parameters as perturbation of the base network.
- [arXiv 2022] Cross-task Attention Mechanism for Dense Multi-task Learning. [paper](https://arxiv.org/abs/2206.08927)
- [AAAI 2023] DeMT: Deformable Mixer Transformer for Multi-Task Learning of Dense Prediction. [paper](https://arxiv.org/abs/2301.03461). [Code](https://github.com/yangyangxu0/DeMT)
  - Leverages the combination of both merits of deformable CNN and query-based Transformer for multi-task learning of dense prediction.
- [ICLR 2023] TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding. [paper](https://openreview.net/forum?id=-CwPopPJda). [Code](https://github.com/prismformore/Multi-Task-Transformer/tree/main/TaskPrompter).
  - a novel spatial-channel multi-task prompting transformer framework.
- [CVPR 2023] Visual Exemplar Driven Task-Prompting for Unified Perception in Autonomous Driving. [paper](https://arxiv.org/abs/2303.01788)
- [arXiv 2023] A Dynamic Feature Interaction Framework for Multi-task Visual Perception. [paper](https://arxiv.org/abs/2306.05061)

<a name='nas'></a>

#### Architecture Search

- [NeurIPS 2020] Adashare: Learning what to share for efficient deep multi-task learning. *ArXiv*, abs/1911.12423, 2020. [Code](https://github.com/sunxm2357/AdaShare)
- [CVPR 2020] Mtl-nas: Task-agnostic neural architecture search towards general-purpose multi-task learning. [Code](https://github.com/bhpfelix/MTLNAS)
- [arXiv 2021] AutoMTL: A Programming Framework for Automated Multi-Task Learning. [paper](https://arxiv.org/abs/2110.13076). [Code](https://github.com/zhanglijun95/AutoMTL)
- [arXiv 2021] FBNetV5: Neural Architecture Search for Multiple Tasks in One Run. [paper](https://arxiv.org/abs/2111.10007)
- [arXiv 2023] AutoTaskFormer: Searching Vision Transformers for Multi-task Learning. [paper](https://arxiv.org/abs/2304.08756)

<a name="dynamic"></a>

#### Dynamic Architecture

- [CVPR 2022] Controllable Dynamic Multi-Task Architectures. [paper](https://arxiv.org/abs/2203.14949). [project](https://www.nec-labs.com/~mas/DYMU/)
- [arXiv 2022] An Evolutionary Approach to Dynamic Introduction of Tasks in Large-scale Multitask Learning Systems. [paper](https://arxiv.org/abs/2205.12755) 
  muNet: Evolving Pretrained Deep Neural Networks into Scalable Auto-tuning Multitask Systems. [paper](https://arxiv.org/abs/2205.10937)
  - Andrea Gesmundo, Jeff Dean
  - A ViT-L architecture (307M params) was evolved into a multitask system with 13087M params jointly solving 69 tasks.
- [NeurIPS 2022] M3ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design. [paper](https://openreview.net/forum?id=cFOhdl1cyU-), [Code](https://github.com/VITA-Group/M3ViT)
  - at training, it disentangles the parameter spaces to avoid different tasks’ training conflicts.
  - at inference, it allows for activating only the task-corresponding sparse “expert” pathway, instead of the full model
- [arXiv 2022] Mod-Squad: Designing Mixture of Experts As Modular Multi-Task Learners. [paper](https://arxiv.org/abs/2212.08066)
  - we incorporate mixture of experts (MoE) layers into a transformer model, with a new loss that incorporates the mutual dependence between tasks and experts. This prevents the sharing of the entire backbone model between all tasks, which strengthens the model, especially when the training set size and the number of tasks scale up.
- [ICLR 2023] Recon: Reducing Conflicting Gradients From the Root For Multi-Task Learning. [paper](https://openreview.net/forum?id=ivwZO-HnzG_)
  - Investigate the task gradients w.r.t. each shared network layer, select the layers with high conflict scores, and set them task-specific. 
- [arXiv 2023] DynaShare: Task and Instance Conditioned Parameter Sharing for Multi-Task Learning. [paper](https://arxiv.org/abs/2305.17305)

<a name="proba"></a>

#### Probabilistic MTL

- [NeurIPS 2021] Variational Multi-Task Learning with Gumbel-Softmax Priors. [paper](https://proceedings.neurips.cc/paper/2021/hash/afd4836712c5e77550897e25711e1d96-Abstract.html)

<a name="trl"></a>
### Task relationship learning

- [CVPR 2018] Taskonomy: Disentangling Task Transfer Learning. 
- [CVPR 2017] Fully-Adaptive Feature Sharing in Multi-Task Networks with Applications in Person Attribute Classification.
- [arXiv 2020] Branched multi-task networks: Deciding what layers to share.
- [arXiv 2020] Automated Search for Resource-Efficient Branched Multi-Task Networks.
- [ICML 2020] Learning to Branch for Multi-Task Learning. [paper](https://arxiv.org/abs/2006.01895v2)
- [arXiv 2020] Measuring and harnessing transference in multi-task learning, [paper](https://arxiv.org/abs/2010.15413v3)
- [ICML 2020] Which Tasks Should Be Learned Together in Multi-task Learning?, http://proceedings.mlr.press/v119/standley20a.html, [Code](https://github.com/tstandley/taskgrouping)
- [ICLR 2021] AUXILIARY TASK UPDATE DECOMPOSITION: THE GOOD, THE BAD AND THE NEUTRAL, https://openreview.net/forum?id=1GTma8HwlYp
  - decompose auxiliary updates into directions which help, damage or leave the primary task loss unchanged
- [NeurIPS 2021] Efficiently Identifying Task Groupings for Multi-Task Learning, [paper](https://arxiv.org/abs/2109.04617), [code](https://github.com/google-research/google-research/tree/master/tag)
  - Our method determines task groupings in a single run by training all tasks together and quantifying the effect to which one task's gradient would affect another task's loss.
  - based on the idea and concepts in *Measuring and harnessing transference in multi-task learning*.
- [arXiv 2022] Editing Models with Task Arithmetic. [paper](https://arxiv.org/abs/2212.04089)

<a name="optim"></a>

### Optimization Methods

<a name="loss"></a>

#### Loss function

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
- [ICRA 2021] OmniDet Surround View Cameras Based Multi-Task Visual Perception Network for Autonomous Driving, [paper](https://arxiv.org/abs/2102.07448)
- [CVPR 2021] Taskology: Utilizing Task Relations at Scale, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Lu_Taskology_Utilizing_Task_Relations_at_Scale_CVPR_2021_paper.html)
- [arXiv 2021] A Closer Look at Loss Weighting in Multi-Task Learning. [paper](https://arxiv.org/abs/2111.10603)
  - MTL model with random weights sampled from a distribution.
- [arXiv 2022] In Defense of the Unitary Scalarization for Deep Multi-Task Learning. [paper](https://arxig.org/abs/2201.04122)
  - None of the ad-hoc multi-task optimization algorithms consistently outperform unitary scalarization, where training simply minimizes the sum of the task losses.
- [ICLR 2022] Weighted Training for Cross-Task Learning, [paper](https://openreview.net/forum?id=ltM1RMZntpu)
  - Target-Aware Weighted Training (TAWT) minimizes a representation-based task distance between the source and target tasks.
- [arXiv 2022] Auto-Lambda: Disentangling Dynamic Task Relationships. [paper](https://arxiv.org/abs/2202.03091), [Code](https://github.com/lorenmt/auto-lambda)
- [arXiv 2022] Universal Representations: A Unified Look at Multiple Task and Domain Learning. [paper](https://arxiv.org/abs/2204.02744), [Code](https://github.com/VICO-UoE/UniversalRepresentations)
  - Distill knowledge from single-task networks.
- [arXiv 2023] Sample-Level Weighting for Multi-Task Learning with Auxiliary Tasks. [paper](https://arxiv.org/abs/2306.04519)

<a name="optimization"></a>

#### Optimization

- [Comptes Rendus Mathematique 2012] Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. [paper](https://www.sciencedirect.com/science/article/pii/S1631073X12000738)
- [NeurIPS 2018] Multi-task learning as multi-objective optimization, [paper](https://arxiv.org/abs/1810.04650)
  - MGDA-UB
- [ICML 2018] Deep asymmetric multi-task feature learning.
- [NeurIPS 2019] Pareto multi-task learning. [paper](https://papers.nips.cc/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html). [Code](https://github.com/xi-l/paretomtl)
- [NeurIPS 2020] Gradient Surgery for Multi-Task Learning. [paper](https://arxiv.org/abs/2001.06782), [Code](https://github.com/tianheyu927/PCGrad)
- [NeurIPS 2020] Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout. [paper](https://arxiv.org/abs/2010.06808)
- [ICML 2020] Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization, [paper](http://proceedings.mlr.press/v119/mahapatra20a.html), [Code](https://github.com/dbmptr/EPOSearch)
- [ICML 2020] Efficient Continuous Pareto Exploration in Multi-Task Learning, [paper](http://proceedings.mlr.press/v119/ma20a.html), [Code](https://github.com/mit-gfx/ContinuousParetoMTL)
- [ICML 2020] Adaptive Adversarial Multi-task Representation Learning, [paper](http://proceedings.mlr.press/v119/mao20a.html)
- [AAAI 2021] Task uncertainty loss reduce negative transfer in asymmetric multi-task feature learning. [paper](https://arxiv.org/abs/2012.09575)
- [AISTATS 2021] High-Dimensional Multi-Task Averaging and Application to Kernel Mean Embedding, [paper](http://proceedings.mlr.press/v130/marienwald21a.html)
- [ICLR 2021] Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models, [paper](https://openreview.net/forum?id=F1vEjWK-lH_)
- [ICLR 2021] Towards Impartial Multi-task Learning, [paper](https://openreview.net/forum?id=IMPnRXEWpvr)
- [NeurIPS 2021] Conflict-Averse Gradient Descent for Multi-task learning, [paper](https://arxiv.org/abs/2110.14048)
- [NeurIPS 2021] Profiling Pareto Front With Multi-Objective Stein Variational Gradient Descent, [paper](https://proceedings.neurips.cc/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html), [code](https://github.com/gnobitab/MultiObjectiveSampling)
- [arXiv 2022] Multi-Task Learning as a Bargaining Game, [paper](https://arxiv.org/abs/2202.01017), [code](https://github.com/AvivNavon/nash-mtl)
- [ICLR 2022] Relational Multi-Task Learning: Modeling Relations between Data and Tasks, [paper](https://openreview.net/forum?id=8Py-W8lSUgy)
  - The proposed MetaLink reinterprets the last layer’s weights of each task as task nodes and creates a knowledge graph where data points and tasks are nodes and labeled edges provide information about labels of data points on tasks.
- [ICLR 2022] RotoGrad: Gradient Homogenization in Multitask Learning, [paper](https://openreview.net/forum?id=T8wHz4rnuGL), [code](https://github.com/adrianjav/rotograd)
  - introduced a rotation layer between the shared backbone and task-specific branches to align gradient directions.
- [ICLR 2022] Sequential Reptile: Inter-Task Gradient Alignment for Multilingual Learning, [paper](https://openreview.net/forum?id=ivQruZvXxtz)
- [arXiv 2022] On Steering Multi-Annotations per Sample for Multi-Task Learning. [paper](https://arxiv.org/abs/2203.02946)
  - Each sample is randomly allocated a subset of tasks during training. (Imo. Can be regarded as a special case of *A Closer Look at Loss Weighting in Multi-Task Learning*.)
- [arXiv 2022] Leveraging convergence behavior to balance conflicting tasks in multi-task learning. [paper](https://arxiv.org/abs/2204.06698)
  - Proposed a method that takes into account temporal behaviour of the gradients to create a dynamic bias that adjust the importance of each task during the backpropagation.
- [arxiv 2022] Do Current Multi-Task Optimization Methods in Deep Learning Even Help? [paper](https://arxiv.org/abs/2209.11379)
  - Despite the added design and computational complexity of these algorithms, MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches.
- [NeurIPS 2022] On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond. [paper](https://openreview.net/forum?id=ScwfQ7hdwyP)
  - The stochastic gradient manipulation algorithms may fail to converge to Pareto optimal solutions and the authors fix it with an algorithm by averaging past weights.
- [AAAI 2023] AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning. [paper](https://arxiv.org/abs/2211.15055)
  - Separate the accumulative gradients and hence the learning rate of each task for each parameter in adaptive learning rate approaches.
- [ICLR 2023] Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach. [paper](https://openreview.net/forum?id=dLAYGdKTi2)
  - a stochastic multi-objective gradient correction (MoCo) method that can guarantee convergence without increasing the batch size even in the nonconvex setting.
- [arXiv 2023] ForkMerge: Overcoming Negative Transfer in Multi-Task Learning. [paper](https://arxiv.org/abs/2301.12618)
- [CVPR 2023] Independent Component Alignment for Multi-Task Learning. [paper](https://arxiv.org/abs/2305.19000), [code](https://github.com/SamsungLabs/MTL)
- [arXiv 2023] FAMO: Fast Adaptive Multitask Optimization. [paper](https://arxiv.org/abs/2306.03792), [code](https://github.com/Cranial-XIX/FAMO)
  - a dynamic weighting method that decreases task losses in a balanced way using O(1) space and time.

<a name='novel'></a>

### Novel Settings

- [CVPR 2019] Deep Virtual Networks for Memory Efficient Inference of Multiple Tasks, [paper](https://ieeexplore.ieee.org/document/8954328/)
- [ICML 2020] Task Understanding from Confusing Multi-task Data, [paper](http://proceedings.mlr.press/v119/su20b.html)
- [ECCV 2020] Multitask Learning Strengthens Adversarial Robustness, [paper](https://arxiv.org/abs/2007.07236v2), [Code](https://github.com/columbia/MTRobust)
- [ICLR 2021] The Traveling Observer Model: Multi-task Learning Through Spatial Variable Embeddings, [paper](https://openreview.net/forum?id=qYda4oLEc1)
  - a machine learning framework in which seemingly unrelated tasks can be solved by a single model, by embedding their input and output variables into a shared space. 
- [arXiv 2021] Learning Multiple Dense Prediction Tasks from Partially Annotated Data, [paper](https://arxiv.org/abs/2111.14893)
- [ICLR 2022] Multi-Task Neural Processes, [paper](https://openreview.net/forum?id=9otKVlgrpZG)
- [DAC 2022] MIME: Adapting a Single Neural Network for Multi-task Inference with Memory-efficient Dynamic Pruning. [paper](https://arxiv.org/abs/2204.05274)
  - MIME results in highly memory-efficient DRAM storage of neural-network parameters for multiple tasks compared to conventional multi-task inference. 
- [T-PAMI 2023] Performance-aware Approximation of Global Channel Pruning for Multitask CNNs, [paper](https://arxiv.org/abs/2303.11923), [Code](http://www.github.com/HankYe/PAGCP.git)
- [arXiv 2023] Efficient Computation Sharing for Multi-Task Visual Scene Understanding, [paper](https://arxiv.org/abs/2303.09663)
- [CVPR 2023] AdaMTL: Adaptive Input-dependent Inference for Efficient Multi-Task Learning. [paper](https://arxiv.org/abs/2304.08594), [Code](https://github.com/scale-lab/AdaMTL.git).

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
- [WoodScape](https://woodscape.valeo.com/): A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving. ICCV 2019. [paper](https://arxiv.org/abs/1905.01489), [code](https://github.com/valeoai/woodscape)
- [TransNAS-Bench-101](https://download.mindspore.cn/dataset/TransNAS-Bench-101): CVPR 2021, Improving Transferability and Generalizability of Cross-Task Neural Architecture Search. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Duan_TransNAS-Bench-101_Improving_Transferability_and_Generalizability_of_Cross-Task_Neural_Architecture_Search_CVPR_2021_paper.html).
- [Omnidata](https://omnidata.vision/): ICCV 2021. Generating multi-task mid-level vision datasets from 3D Scans. [paper](https://arxiv.org/abs/2110.04994). 

<a name="apps"></a>
## Applications

#### Natural language processing

- [ICML 2008] A unified architecture for natural language processing: deep neural networks with multitask learning, https://dl.acm.org/doi/10.1145/1390156.1390177
- [ICLR 2021] Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data, https://openreview.net/pdf?id=de11dbHzAMF, [Code](https://github.com/CAMTL/CA-MTL)
- [ICLR 2021] HyperGrid Transformers: Towards A Single Model for Multiple Tasks, https://openreview.net/forum?id=hiq1rHO8pNT
- [ICML 2020] XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation, http://proceedings.mlr.press/v119/hu20b.html, [Code](https://github.com/google-research/xtreme)

#### Speech processing

#### Computer vision

##### Medical Imaging

- [MIDL 2020] Extending Unsupervised Neural Image Compression With Supervised Multitask Learning, http://proceedings.mlr.press/v121/tellez20a.html

##### Autonomous Driving

- [ICML 2019] Multi-task learning in the wildness. https://slideslive.com/38917690/multitask-learning-in-the-wilderness
- [arXiv 2020] Efficient Latent Representations using Multiple Tasks for Autonomous Driving, [paper](https://arxiv.org/abs/2003.00695)
- [arXiv 2021] MonoGRNet: A General Framework for Monocular 3D Object Detection, [paper](https://arxiv.org/abs/2104.08797)
- [CVPR 2021] Multi-task Learning with Attention for End-to-end Autonomous Driving. [paper](https://arxiv.org/abs/2104.10753)
- [ICRA 2021] OmniDet Surround View Cameras Based Multi-Task Visual Perception Network for Autonomous Driving, [paper](https://arxiv.org/abs/2102.07448)
- [CVPR 2021] Deep Multi-Task Learning for Joint Localization, Perception, and Prediction. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Phillips_Deep_Multi-Task_Learning_for_Joint_Localization_Perception_and_Prediction_CVPR_2021_paper.html)
- [CVPR 2022 workshop] LidarMultiNet: Unifying LiDAR Semantic Segmentation, 3D Object Detection, and Panoptic Segmentation in a Single Multi-task Network. [paper](https://arxiv.org/abs/2206.11428)
  - LidarMultiNet: Towards a Unified Multi-task Network for LiDAR Perception. [paper](https://arxiv.org/abs/2209.09385)
- [arXiv 2022] BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving. [paper](https://arxiv.org/abs/2205.09743). [code](https://github.com/zhangyp15/BEVerse)
- [arXiv 2022] Perceive, Interact, Predict: Learning Dynamic and Static Clues for End-to-End Motion Prediction. [paper](https://arxiv.org/abs/2212.02181)
  -  jointly and interactively performs online mapping, object detection and motion prediction.
- [IEEE TIV 2022] Surround-view Fisheye BEV-Perception for Valet Parking: Dataset, Baseline and Distortion-insensitive Multi-task Framework. [paper](https://arxiv.org/abs/2212.04111)
- [arXiv 2022] **Goal-oriented Autonomous Driving.** [paper](https://arxiv.org/abs/2212.10156). [project](https://opendrivelab.github.io/UniAD/)
  - the first comprehensive framework up-to-date that incorporates full-stack driving tasks in one network. 
- [arXiv 2023] AOP-Net: All-in-One Perception Network for Joint LiDAR-based 3D Object Detection and Panoptic Segmentation. [paper](https://arxiv.org/abs/2302.00885)
- [arXiv 2023] LiDARFormer: A Unified Transformer-based Multi-task Network for LiDAR Perception. [paper](https://arxiv.org/abs/2303.12194)
- [CVPR 2023] TBP-Former: Learning Temporal Bird's-Eye-View Pyramid for Joint Perception and Prediction in Vision-Centric Autonomous Driving. [paper](https://arxiv.org/abs/2303.09998). [code](https://github.com/MediaBrain-SJTU/TBP-Former).

##### Others

- [SIGKDD 2019] Learning a Unified Embedding for Visual Search at Pinterest, [paper](https://arxiv.org/abs/1908.01707v1)
  - For every mini-batch, we **balance a uniform mix of each of the datasets** with an epoch defined by the iterations to iterate through the largest dataset. Each dataset has its own indepedent tasks so we **ignore the gradient contributions of images on tasks that it does not have data for**. The losses from all the tasks are assigned equal weights and are summed for backward propagation.
- [CVPR 2021] When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_When_Age-Invariant_Face_Recognition_Meets_Face_Age_Synthesis_A_Multi-Task_CVPR_2021_paper.html), [Code](https://github.com/Hzzone/MTLFace)
- [CVPR 2021] Three Birds with One Stone: Multi-Task Temporal Action Detection via Recycling Temporal Annotations, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Three_Birds_with_One_Stone_Multi-Task_Temporal_Action_Detection_via_CVPR_2021_paper.html)
- [CVPR 2021] Anomaly Detection in Video via Self-Supervised and Multi-Task Learning, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html)

#### Reinforcement learning

- [arXiv 2021] MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale, https://arxiv.org/abs/2104.08212
- [ICML 2020] CoMic: Complementary Task Learning & Mimicry for Reusable Skills, http://proceedings.mlr.press/v119/hasenclever20a.html
- [AISTATS 2021] On the Effect of Auxiliary Tasks on Representation Dynamics, http://proceedings.mlr.press/v130/lyle21a.html
- [ICML 2021] Multi-Task Reinforcement Learning with Context-based Representations, [paper](https://arxiv.org/abs/2102.06177)

#### Recommendation

- [AISTATS 2021] Decision Making Problems with Funnel Structure: A Multi-Task Learning Approach with Application to Email Marketing Campaigns, http://proceedings.mlr.press/v130/xu21a.html

#### Multi-modal

- [CVPR 2014] https://sites.google.com/site/deeplearningcvpr2014/DL-Multimodal_multitask_learning.pdf Multimodal learning and multitask learning 
- [arXiv 2020] Multimodal Continuous Emotion Recognition using Deep Multi-Task Learning with Correlation Loss, https://arxiv.org/abs/2011.00876
- [arXiv 2021] Towards General Purpose Vision Systems, https://arxiv.org/pdf/2104.00743.pdf
- [arXiv 2021] UniT: Multimodal Multitask Learning with a Unified Transformer, https://arxiv.org/abs/2102.10772, [Code](https://mmf.sh/)
- [NeurIPS 2021] Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning, [paper](https://arxiv.org/abs/2110.14202), [code](https://miladabd.github.io/KML)
  - quantitative, task-level analysis inspired by the recent transference idea from multi-task learning
- [NeurIPS 2021] Referring Transformer: A One-step Approach to Multi-task Visual Grounding, [paper](https://arxiv.org/abs/2106.03089)
- [arXiv 2022] MultiMAE: Multi-modal Multi-task Masked Autoencoders. [project](https://multimae.epfl.ch/), [paper](https://arxiv.org/abs/2204.01678), [code](https://github.com/EPFL-VILAB/MultiMAE)
- [arXiv 2022] OFASys: A Multi-Modal Multi-Task Learning System for Building Generalist Models. [paper](https://arxiv.org/abs/2212.04408), [Code](https://github.com/OFA-Sys/OFASys)
- [ICLR 2023] UNIFIED-IO: A Unified Model for Vision, Language, and Multi-modal Tasks. [paper](https://openreview.net/forum?id=E01k9048soZ)

<a name="related"></a>

## Related Areas

- Transfer Learning
  - [CVPR 2021] Can We Characterize Tasks Without Labels or Features? [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wallace_Can_We_Characterize_Tasks_Without_Labels_or_Features_CVPR_2021_paper.html), [code](https://github.com/BramSW/)
  - [CVPR 2021] OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations, [paper](https://arxiv.org/abs/2103.13843)
- Auxiliary Learning
  - [CVPR 2021] Image Change Captioning by Learning From an Auxiliary Task, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Hosseinzadeh_Image_Change_Captioning_by_Learning_From_an_Auxiliary_Task_CVPR_2021_paper.html)
- Multi-label Learning
  - [AAAI 2023] Incomplete Multi-View Multi-Label Learning via Label-Guided Masked View- and Category-Aware Transformers. [paper](https://arxiv.org/abs/2303.07180)
- Multi-modal Learning
- Meta Learning
- Continual Learning
  - [CVPR 2021] KSM: Fast Multiple Task Adaption via Kernel-wise Soft Mask Learning, [paper](https://arxiv.org/abs/2009.05668)
  - [ICLR 2021] Linear Mode Connectivity in Multitask and Continual Learning, https://openreview.net/forum?id=Fmg_fQYUejf, [Code](https://github.com/imirzadeh/MC-SGD)
  - [arXiv 2023] SubTuning: Efficient Finetuning for Multi-Task Learning. [paper](https://arxiv.org/abs/2302.06354)
- Curriculum Learning
- Ensemble, Distillation and Model Fusion
- Federal Learning
  - [NeurIPS 2021] Federated Multi-Task Learning under a Mixture of Distributions, [paper](https://arxiv.org/abs/2108.10252)
  - [arXiv 2022] Multi-Task Distributed Learning using Vision Transformer with Random Patch Permutation. [paper](https://arxiv.org/abs/2204.03500)
- Active Learning
  - [arXiv 2022] PartAL: Efficient Partial Active Learning in Multi-Task Visual Settings, [paper](https://arxiv.org/abs/2211.11546)

<a name="trends"></a>

# Trends

- [Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/): multi-task, multi-modal, sparse activated
