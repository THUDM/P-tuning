import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from os.path import join, abspath, dirname

from LAMA.data_utils.dataset import load_file, LAMADataset
from LAMA.data_utils.vocab import init_vocab
from LAMA.p_tuning.modeling import PTuneForLAMA

SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large',
                  'megatron_11b']


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="P1001")
    parser.add_argument("--model_name", type=str, default='megatron_11b', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(3, 3, 3)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), '../data/LAMA'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), '../out/LAMA'))
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()

    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:0' if self.args.model_name != 't5-11b' else 'cuda:{}'.format(self.args.t5_shard * 4)

        if self.args.use_original_template and (not self.args.use_lm_finetune) and (not self.args.only_evaluate):
            raise RuntimeError("""If use args.use_original_template is True, 
            either args.use_lm_finetune or args.only_evaluate should be True.""")

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)

        # load datasets and dataloaders
        self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()

        self.train_data = load_file(join(self.args.data_dir, self.data_path_pre + 'train' + self.data_path_post))
        self.dev_data = load_file(join(self.args.data_dir, self.data_path_pre + 'dev' + self.data_path_post))
        self.test_data = load_file(join(self.args.data_dir, self.data_path_pre + 'test' + self.data_path_post))

        self.test_set = LAMADataset('test', self.test_data, self.tokenizer, self.args)
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer, self.args)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer, self.args)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(self.train_set, batch_size=8, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8)
        self.test_loader = DataLoader(self.test_set, batch_size=8)

        self.model = PTuneForLAMA(args, self.device, self.args.template)

    def get_TREx_parameters(self):
        relation = load_file(join(self.args.data_dir, "single_relations/{}.jsonl".format(self.args.relation_id)))[0]
        data_path_pre = "fact-retrieval/original/{}/".format(self.args.relation_id)
        data_path_post = ".jsonl"
        return relation, data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss = 0, 0
            for x_hs, x_ts in loader:
                if False and self.args.extend_data:
                    _loss, _hit1 = self.model.test_extend_data(x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, top10 = self.model(x_hs, x_ts, return_candidates=True)
                else:
                    _loss, _hit1 = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
            hit1 /= len(dataset)
            print("{} {} Epoch {} Loss: {} Hit@1:".format(self.args.relation_id, evaluate_type, epoch_idx,
                                                          loss / len(dataset)), hit1)
        return loss, hit1

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# Prompt:", self.model.prompt)
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(100):
            # check early stopping
            if epoch_idx > -1:
                dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'Dev')
                if epoch_idx == 0:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                if epoch_idx > 0 and (dev_hit1 >= best_dev) or self.args.only_evaluate:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                    best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1)
                    early_stop = 0
                    best_dev = dev_hit1
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        self.save(best_ckpt)
                        print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt
            if self.args.only_evaluate:
                break

            # run training
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1 = self.model(batch[0], batch[1])
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt


def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.relation_id)
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
