# Token-to-Token-ViT-flax

"Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, e.g., the Vision Transformer (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance to CNNs when trained from scratch on a midsize dataset like ImageNet. We find it is because: 1) the simple tokenization of input images fails to model the important local structure such as edges and lines among neighboring pixels, leading to low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness for fixed computation budgets and limited training samples. To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study. Notably, T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0\% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets by directly training on ImageNet. For example, T2T-ViT with comparable size to ResNet50 (21.5M parameters) can achieve 83.3\% top1 accuracy in image resolution 384×384 on ImageNet." - Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan

## Acknowledgement:
I have been greatly inspired by the work of [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

## Usage:
```python
import numpy as np

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 224, 224, 3))

v = T2TViT(
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")
```

## Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

## Citations:
```bibtex
@article{DBLP:journals/corr/abs-2101-11986,
  author    = {Li Yuan and
               Yunpeng Chen and
               Tao Wang and
               Weihao Yu and
               Yujun Shi and
               Francis E. H. Tay and
               Jiashi Feng and
               Shuicheng Yan},
  title     = {Tokens-to-Token ViT: Training Vision Transformers from Scratch on
               ImageNet},
  journal   = {CoRR},
  volume    = {abs/2101.11986},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.11986},
  eprinttype = {arXiv},
  eprint    = {2101.11986},
  timestamp = {Mon, 04 Apr 2022 16:15:35 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-11986.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```