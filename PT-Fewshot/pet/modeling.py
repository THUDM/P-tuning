# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import statistics
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
from pet.config import EvalConfig, TrainConfig
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper
from pet.config import  WrapperConfig

logger = log.get_logger('root')




def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_pet(train_data: List[InputExample],
              eval_data: List[InputExample],
              dev32_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            # Training
            if do_train:

                results_dict.update(train_single_model(train_data, eval_data, dev32_data, pattern_iter_output_dir, \
                                                       wrapper, train_config, eval_config))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if not do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info("Starting evaluation...")

                # if not wrapper:
                wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                eval_result = evaluate(wrapper, eval_data, eval_config)
                dev32_result = evaluate(wrapper, dev32_data, eval_config)

                save_predictions(os.path.join(pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

                logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info("eval_results:")
                logger.info(eval_result['scores'])
                logger.info("dev32_results:")
                logger.info(dev32_result['scores'])

                results_dict['eval_set_after_training'] = eval_result['scores']
                results_dict['dev32_set_after_training'] = dev32_result['scores']
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in eval_result['scores'].items():
                    results[metric][pattern_id].append(value)

                for metric, value in dev32_result['scores'].items():
                    dev32_results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev32_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev32_data=dev32_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
    return results_dict


def evaluate(model: TransformerModelWrapper,
             eval_data: List[InputExample],
             config: EvalConfig) -> Dict:

    metrics = config.metrics if config.metrics else ['acc']
    results = model.eval(eval_data=eval_data,
                         per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu)
    predictions = np.argmax(results['logits'], axis=1)
    scores = {}
    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")
    results['scores'] = scores
    results['predictions'] = predictions
    return results


def _write_results(path: str, all_results: Dict, dev32_results: Dict):
    with open(path, 'w') as fh:

        results = all_results
        logger.info("eval_results:")
        fh.write("eval_results:" + '\n')

        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        logger.info("dev32_results:")
        fh.write("dev32_results:" + '\n')

        for metric in dev32_results.keys():
            for pattern_id, values in dev32_results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in dev32_results.keys():
            all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

