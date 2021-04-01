# P-tuning for LAMA
## Usage
run cli.py to start the experiment
```bash
python cli.py
```

## Data
We adopt the dataset created by [AutoPrompt](https://github.com/ucinlp/autoprompt), and created the shared vocab which removes 
some stopwords using the scripts from original [LAMA](https://github.com/facebookresearch/LAMA). The packed up data file is available. 
[[THU Cloud Drive]](https://cloud.tsinghua.edu.cn/f/21b9dcf05cc44adfad25/?dl=1)

If you use our packed up data, please download it and unzip it in the *data/* folder in the root directory.

## About MegatronLM (11B)
The original model checkpoint is available in [FairSeq](https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b), 
which applies the Megatron for model parallel and need at least 8 V100 GPUs. 

In our experiment, we freeze the parameters of MegatronLM (11B) and only train the continuous prompt, and thus merge the 
splited 8 model partitions into one and load into a 32G V100 GPU. We provide the merge function in ./megatron_11b/megatron_wrapper.py. 
If you want to use the model parallel feature, please refer to  the implemention in FairSeq and Megatron.

You can create a *checkpoints/* folder in the root directory.
