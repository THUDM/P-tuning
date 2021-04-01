# P-Tuning for Few-shot


### Usage

The few-shot experiment results can be reproduced by running scripts as follows.

```bash
sh scripts/rte_pt_few_shot.sh
```


### Data

The [FewGLUE_32dev](https://github.com/THUDM/P-tuning/tree/main/FewGLUE_32dev) dataset is adopted for experiments.
[PT-Fewshot](https://github.com/THUDM/P-tuning/tree/main/PT-Fewshot/data_utils) contains utilities for loading, preprocessing and pattern-verbalizer transformation of FewGLUE_32dev data.


### Tips

We have summarized several empirical observations that might guide further exploration of P-Tuning.

1. The position of prompt tokens and anchors matter.
   
2. The order of few-shot training data affects a lot.

3. Choosing a larger learning rate particularly for prompt embeddings leads to better performance.


### Acknowledgement

The code is developed based on [pet](https://github.com/timoschick/pet).
We appreciate all the authors who made their code public, which greatly facilitates this project. 
This repository would be continuously updated.
