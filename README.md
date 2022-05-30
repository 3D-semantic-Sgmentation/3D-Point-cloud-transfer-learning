# Point-cloud-transfer-learning
This repository collects code, videos related to point cloud transfer learning

## Self-Supervised Learning for Domain Adaptation on Point-Clouds
Self-supervised learning (SSL) allows to learn useful representations from unlabeled data and has been applied effectively for domain adaptation (DA) on images. It is still unknown if and how it can be leveraged for domain adaptation for 3D perception. Here we describe the first study of SSL for DA on point clouds. We introduce a new family of pretext tasks, Deformation Reconstruction, motivated by the deformations encountered in sim-to-real transformations. The key idea is to deform regions of the input shape and use a neural network to reconstruct them. We design three types of shape deformation methods: (1) Volume-based: shape deformation based on proximity in the input space; (2) Feature-based: deforming regions in the shape that are semantically similar; and (3) Sampling-based: shape deformation based on three simple sampling schemes. As a separate contribution, we also develop a new method based on the Mixup training procedure for point-clouds. Evaluations on six domain adaptations across synthetic and real furniture data, demonstrate large improvement over previous work.
[[Paper](https://arxiv.org/pdf/2003.12641.pdf)] [[code](https://github.com/IdanAchituve/DefRec_and_PCM)]

## LiDARNet: A Boundary-Aware Domain Adaptation Model for Point Cloud Semantic Segmentation
We present a boundary-aware domain adaptation model for LiDAR scan full-scene semantic segmentation (LiDARNet). Our model can extract both the domain private features and the domain shared features with a two branch structure. We embedded Gated-SCNN into the segmentor component of LiDARNet to learn boundary information while learning to predict full-scene semantic segmentation labels. Moreover, we further reduce the domain gap by inducing the model to learn a mapping between two domains using the domain shared and private features. Besides, we introduce a new dataset (SemanticUSL). The dataset has the same data format and ontology as SemanticKITTI. We conducted experiments on real-world datasets SemanticKITTI, SemanticPOSS, and SemanticUSL, which have differences in channel distributions, reflectivity distributions, diversity of scenes, and sensors setup. Using our approach, we can get a single projection-based LiDAR full-scene semantic segmentation model working on both domains. Our model can keep almost the same performance on the source domain after adaptation and get an 8%-22% mIoU performance increase in the target domain.
[[paper](https://arxiv.org/abs/2003.01174)][[code](https://github.com/unmannedlab/LiDARNet)come soon]


### 2D GAN Video
[[Youtube Channel](https://www.youtube.com/watch?v=UcHe0xiuvpg&list=RDCMUC34rW-HtPJulxr5wp2Xa04w&index=10)]

### 2D GAN Code
[[Keras-GAN Code](https://github.com/eriklindernoren/Keras-GAN)]
[[Pytorch-GAN Code](https://github.com/eriklindernoren/PyTorch-GAN)]
