# P-tuning
A novel method to tune language models. Codes and datasets for paper [``GPT understands, too''](https://arxiv.org/abs/2103.10385).

![](img/PT.png)
## How to use our code
We have released the code and datasets for LAMA and few-shot SuperGLUE (32-dev) experiments. Please check **README.md** and **requirement.txt** in the corresponding subdirectories for details.

The [LAMA](https://cloud.tsinghua.edu.cn/f/21b9dcf05cc44adfad25/?dl=1) and [few-shot SuperGLUE (32-dev)](https://cloud.tsinghua.edu.cn/f/526f471aed544d248949/?dl=1) datasets are available. The LAMA dataset should be placed in ./data directory, and the SuperGLUE dataset should be placed in the ./ (project root) directory.

## Citation

If you find our work useful, please cite the following paper:

    @article{liu2021gpt,
      title={GPT Understands, Too}, 
      author={Xiao Liu and Yanan Zheng and Zhengxiao Du and Ming Ding and Yujie Qian and Zhilin Yang and Jie Tang},
      year={2021},
      journal={arXiv preprint arXiv:2103.10385},
      url={https://arxiv.org/abs/2103.10385}
    }
