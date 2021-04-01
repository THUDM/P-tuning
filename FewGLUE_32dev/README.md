# FewGLUE_32dev

This repository contains the FewGLUE_32dev dataset, an extension of the [FewGLUE](https://github.com/timoschick/fewglue), which enables NLU few-shot learning tasks to be benchmarked under a new 32-sample-dev setting. It has been proved in [previous work](https://arxiv.org/abs/2012.15723) that using larger development sets confer a significant advantage beyond few-shot. FewGLUE_32dev is built by adding additional few-shot dev sets with 32 examples randomly selected from the original/unused SuperGLUE training sets.


### Data Format

The data files follow the exact same format as [SuperGLUE task files](https://super.gluebenchmark.com/tasks).


### Structure

For each SuperGLUE task `T`, the directory `FewGLUE_32dev/T` contains the 32-sample-dev file (`dev32.jsonl`), which consists of 32 examples for few-shot validation.

To perform few-shot learning under 32-dev setting, the following files are also required, including the FewGLUE train files (`train.jsonl`)[[download](https://github.com/timoschick/fewglue)], and the SuperGLUE validation/test files (`val.jsonl`/`test.jsonl`)[[download](https://super.gluebenchmark.com/tasks)].
