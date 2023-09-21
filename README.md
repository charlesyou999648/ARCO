# [NeurIPS'23] Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

[![arXiv](https://img.shields.io/badge/arXiv-2302.01735-b31b1b.svg)](https://arxiv.org/abs/2302.01735)

This is the PyTorch implemention of our NeurIPS 2023 paper "**Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective**" by [Chenyu You](http://chenyuyou.me/), [Weicheng Dai](https://weichengdai1.github.io/), [Yifei Min](https://scholar.google.com/citations?user=pFWnzL0AAAAJ&hl=en/), [Fenglin Liu](https://eng.ox.ac.uk/people/fenglin-liu/), [David A. Clifton](https://eng.ox.ac.uk/people/david-clifton/), [S. Kevin Zhou](https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en/), [Lawrence Staib](https://medicine.yale.edu/profile/lawrence-staib/), and [James S. Duncan](https://medicine.yale.edu/profile/james-duncan/). 

## Abstract
> For medical image segmentation, contrastive learning is the dominant practice to improve the quality of visual representations by contrasting semantically similar and dissimilar pairs of samples. This is enabled by the observation that without accessing ground truth labels, negative examples with truly dissimilar anatomical features, if sampled, can significantly improve the performance. In reality, however, these samples may come from similar anatomical regions and the models may struggle to distinguish the minority tail-class samples, making the tail classes more prone to misclassification, both of which typically lead to model collapse. In this paper, we propose ARCO, a semi-supervised contrastive learning (CL) framework with stratified group theory for medical image segmentation. In particular, we first propose building ARCO through the concept of variance-reduced estimation and show that certain variance-reduction techniques are particularly beneficial in pixel/voxel-level segmentation tasks with extremely limited labels. Furthermore, we theoretically prove these sampling techniques are universal in variance reduction. Finally, we experimentally validate our approaches on eight benchmarks, i.e., five 2D/3D medical and three semantic segmentation datasets, with different label settings, and our methods consistently outperform state-of-the-art semi-supervised methods. Additionally, we augment the CL frameworks with these sampling techniques and demonstrate significant gains over previous methods. We believe our work is an important step towards semi-supervised medical image segmentation by quantifying the limitation of current self-supervision objectives for accomplishing such challenging safety-critical tasks.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{you2023rethinking,
  title={Rethinking semi-supervised medical image segmentation: A variance-reduction perspective},
  author={You, Chenyu and Dai, Weicheng and Min, Yifei and Liu, Fenglin and Clifton, David A and Zhou, S Kevin and Staib, Lawrence Hamilton and Duncan, James S},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
