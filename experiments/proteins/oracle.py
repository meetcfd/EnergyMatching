from typing import List, Optional, Tuple
from Levenshtein import distance as levenshtein
import numpy as np
import logging
import os
import pandas as pd
from utils_proteins import Encoder
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alphabet = "ARNDCQEGHILKMFPSTWYV"

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0*x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class BaseCNN(nn.Module):
    def __init__(
            self,
            n_tokens: int = 20,
            kernel_size: int = 5 ,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True,
            activation: str = 'relu',
            linear: bool=True,
            **kwargs):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) 
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        #onehotize
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        return output

    def forward_soft(self, x):
        x = torch.softmax(x,-1)
        # encoder
        x = self.encoder(x.permute(0, 2, 1)).permute(0,2,1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        return output

def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

def _read_fasta(fasta_path):
    fasta_seqs = fasta.FastaFile.read(fasta_path)
    seq_to_fitness = {}
    process_header = lambda x: float(x.split('_')[-1].split('=')[1])
    for x,y in fasta_seqs.items():
        seq_to_fitness[y] = process_header(x)
    return seq_to_fitness

class EvalRunner:
    def __init__(self, scenario, runner_cfg):
        self._cfg = runner_cfg
        self._log = logging.getLogger(__name__)
        self.predictor_tokenizer = Encoder()
        gt_csv = pd.read_csv(self._cfg.gt_csv)
        oracle_dir = self._cfg.oracle_dir
        self.use_normalization = self._cfg.use_normalization
        # Read in known sequences and their fitnesses
        self._max_known_score = np.max(gt_csv.score)
        self._min_known_score = np.min(gt_csv.score)
        self.normalize = lambda x: to_np((x - self._min_known_score) / (self._max_known_score - self._min_known_score)).item()
        self._log.info(f'Read in {len(gt_csv)} ground truth sequences.')
        self._log.info(f'Maximum known score {self._max_known_score}.')
        self._log.info(f'Minimum known score {self._min_known_score}.')

        # Read in base pool used to generate sequences.
        base_pool_seqs = pd.read_csv(self._cfg.base_pool_path)
        self._base_pool_seqs = base_pool_seqs.sequence.tolist()
        self.device = torch.device('cuda') #requires GPU
        oracle_path = os.path.join(oracle_dir, 'oracle_' + str(scenario) + '.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=self.device)
        cfg_path = os.path.join(oracle_dir, 'config_' + str(scenario) + '.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)

        task = 'medium' if 'medium' in runner_cfg.base_pool_path else 'hard'

        self._cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) # oracle has same architecture as predictor
        self._cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        self._cnn_oracle = self._cnn_oracle.to(self.device)
        self._cnn_oracle.eval()
        if self._cfg.predictor_dir is not None:
            predictor_path = os.path.join(self._cfg.predictor_dir, 'predictor_' + str(scenario) + '_' + str(task) + '.ckpt')
            predictor_state_dict = torch.load(predictor_path, map_location=self.device)
            self._predictor = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
            self._predictor.load_state_dict(
                {k.replace('predictor.', ''): v for k,v in predictor_state_dict['state_dict'].items()})
            self._predictor = self._predictor.to(self.device)
        self.run_oracle = self._run_cnn_oracle
        self.run_predictor = self._run_predictor if self._cfg.predictor_dir is not None else None


    def novelty(self, sampled_seqs):
        # sampled_seqs: top k
        # existing_seqs: range dataset
        all_novelty = []
        for src in tqdm(sampled_seqs):  
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty

    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)

    def _run_cnn_oracle(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._cnn_oracle(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)

    def _run_predictor(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._predictor(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)
    
    def evaluate_sequences(self, topk_seqs, use_oracle = True):
        topk_seqs = list(set(topk_seqs))
        num_unique_seqs = len(topk_seqs)
        topk_scores = self.run_oracle(topk_seqs) if use_oracle else self.run_predictor(topk_seqs)
        normalized_scores = [self.normalize(x) for x in topk_scores]
        seq_novelty = self.novelty(topk_seqs)
        results_df = pd.DataFrame({
            'sequence': topk_seqs,
            'oracle_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })  if use_oracle else pd.DataFrame({
            'sequence': topk_seqs,
            'predictor_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })

        if num_unique_seqs == 1:
            seq_diversity = 0
        else:
            seq_diversity = diversity(topk_seqs)
               
        metrics_scores = normalized_scores if self.use_normalization else topk_scores.detach().cpu().numpy()
        metrics_df = pd.DataFrame({
            'num_unique': [num_unique_seqs],
            'mean_fitness': [np.mean(metrics_scores)],
            'mean_fitness': [np.mean(metrics_scores)],
            'median_fitness': [np.median(metrics_scores)],
            'std_fitness': [np.std(metrics_scores)],
            'max_fitness': [np.max(metrics_scores)],
            'mean_diversity': [seq_diversity],
            'mean_novelty': [np.mean(seq_novelty)],
            'median_novelty': [np.median(seq_novelty)],
        })
        return results_df, metrics_df

def process_ggs_seqs(samples_path, sampling_method, topk, epoch_filter):
    """Process ggs samples."""
    generated_pairs = pd.read_csv(samples_path)
    print(len(generated_pairs.mutant_sequence.unique()))
    generated_pairs = generated_pairs.drop_duplicates(
        subset='mutant_sequence', keep = 'first', ignore_index=True)
   
    print(generated_pairs.shape)
    
    if epoch_filter is not None:
        if epoch_filter == 'last':
            generated_pairs = generated_pairs[generated_pairs.epoch == generated_pairs.epoch.max()]
        else:
            raise ValueError(f'Bad epoch filter: {epoch_filter}')
    
    print(generated_pairs.shape)
    if sampling_method == 'greedy':
        generated_pairs = generated_pairs.sort_values(
            'mutant_score', ascending=False)
        sampled_seqs = generated_pairs.mutant_sequence.tolist()[:topk]
        log.info(f'Sampled {len(set(sampled_seqs))} unique sequences.')
    else:
        raise ValueError(f'Bad sampling method: {sampling_method}')
    return sampled_seqs

def process_baseline_seqs(samples_path, topk):
    """Process baseline samples."""
    df = pd.read_csv(samples_path)
    column_name = 'sequence' if 'sequence' in df.columns else df.columns[0]
    sampled_seqs = df[column_name].tolist()
    return sampled_seqs

def process_mc_seqs(samples_matrix_path, fitness_matrix_path, topk):
    samples_matrix = pd.read_csv(samples_matrix_path)
    fitness_matrix = pd.read_csv(fitness_matrix_path)
    last_column = samples_matrix.iloc[:, 8]
    top_indices = fitness_matrix.iloc[:, 8].nlargest(topk).index
    top_seqs = last_column.iloc[top_indices].tolist()
    return top_seqs

def eval(scenario, task, baselines_samples_dir, use_pred=False):
    gt_file = scenario.lower() + "_ground_truth.csv" 
    base_pool_path = os.path.join(os.getcwd(),'data', scenario.lower() + "_" + task + ".csv")

    topk = None 
    results_dir = "/".join(baselines_samples_dir.split('/')[:-1])

    samples_dir = baselines_samples_dir
    _method_fn = lambda x: process_baseline_seqs(x, topk)
    
    cfg = OmegaConf.create({
        "experiment": {
            "gt_csv": os.path.join(os.getcwd(),'data', gt_file), 
            "oracle_dir": os.path.join(os.getcwd(),'oracle'),
            "predictor_dir": os.path.join(os.getcwd(),'predictor','unsmoothed') 
        },
        "runner": {
            "batch_size": 128,
            "base_pool_path": base_pool_path,
            "oracle": "cnn",
            "gt_csv": "${experiment.gt_csv}",
            "oracle_dir": "${experiment.oracle_dir}",
            "predictor_dir": "${experiment.predictor_dir}",
            "use_normalization": True
        }
    })

    eval_runner = EvalRunner(scenario, cfg.runner)

    # Glob results to evaluate.
    all_csv_paths = [samples_dir]

    # Run evaluation for each result.
    all_results = []
    all_metrics = []
    all_acceptance_rates = []
    use_oracle = True if not use_pred else False

    if '-MC' in samples_dir:
        # If the directory contains '-MC', process the matrices instead of CSVs
        matrix_files = glob.glob(os.path.join(samples_dir, 'samples_matrix_seed_*.csv'))
        for matrix_file in tqdm(matrix_files):
            seed = matrix_file.split('_')[-1].split('.')[0]  # Extract seed from filename
            samples_matrix_path = matrix_file
            fitness_matrix_path = os.path.join(samples_dir, f'fitness_matrix_seed_{seed}.csv')
            topk_seqs = _method_fn(samples_matrix_path, fitness_matrix_path)  # Process the matrices and get topk sequences
            csv_results, csv_metrics = eval_runner.evaluate_sequences(topk_seqs, use_oracle=use_oracle)
            log.info(f'Results for {matrix_file}\n{csv_metrics}')
            csv_results['source_path'] = matrix_file
            csv_metrics['source_path'] = matrix_file
            all_results.append(csv_results)
            all_metrics.append(csv_metrics)
    else:
        # Existing loop for processing CSVs
        for csv_path in tqdm(all_csv_paths):
            # csv_path = os.path.join(results_dir, csv_path)
            topk_seqs = _method_fn(csv_path)
            csv_results, csv_metrics = eval_runner.evaluate_sequences(topk_seqs, use_oracle=use_oracle)
            # log.info(f'Results for {csv_path}\n{csv_metrics}')
            csv_results['source_path'] = csv_path
            csv_metrics['source_path'] = csv_path
            all_results.append(csv_results)
            all_metrics.append(csv_metrics)

    all_results = pd.concat(all_results) 
    all_metrics = pd.concat(all_metrics)

    print(all_metrics)
    print('\nmedian fitness: %.4f' %all_results['normalized_score'].median())
    print('diversity: ', all_metrics['mean_diversity'][0].item())
    print('novelty: ', all_metrics['median_novelty'][0].item())

    return all_results['normalized_score'].median(), all_metrics['mean_diversity'][0].item(), all_metrics['median_novelty'][0].item()

  