import sys
sys.path.append('/work/jiaqi/ppi-mut')
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
from Bio import SeqIO
import os, time, json, argparse
from tqdm import tqdm
from scipy.stats import spearmanr

import esm
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter

from utils import commons

def parse_chain_ids(pdb):
    parts = pdb.strip().split('_')
    chain_ids = []
    for i in range(1, len(parts)):
        chain_ids.extend(list(parts[i]))
    return chain_ids

def concatenate_coords(coords, padding_length=10):
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)

    coords_list = []
    ranges = {}
    pos = 0

    for i, chain_id in enumerate(coords):
        if i > 0:
            coords_list.append(pad_coords)
            pos += padding_length
        L = coords[chain_id].shape[0]
        coords_list.append(coords[chain_id])
        ranges[chain_id] = (pos, pos + L)
        pos += L

    return np.concatenate(coords_list, axis=0), ranges

def concatenate_seqs(seqs, alphabet, ranges, padding_length=10, mask_token='<mask>'):
    seqs_list = []
    for i, chain_id in enumerate(ranges):
        if i > 0:
            seqs_list.extend([mask_token] * padding_length)
        seqs_list.extend(list(seqs[chain_id]))
    return ''.join(seqs_list)

def get_logits(model, alphabet, coords, seq):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)
    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    return logits

def naive_score(model, alphabet, fpath, chain_ids, mut, logger):
    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    concat_coords, ranges = concatenate_coords(coords)
    concat_seq = concatenate_seqs(native_seqs, alphabet, ranges)
    logits = get_logits(model, alphabet, concat_coords, concat_seq)
    # logger.info(f'{pdb} logits shape: {logits.shape}') # [1, 35, 347], batch_size x vocab_size x seq_len
    logits = logits.permute(0, 2, 1) # [1, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1) # [1, seq_len, vocab_size]
    log_probs = log_probs.squeeze(0) # [seq_len, vocab_size]
    # logger.info(f'{pdb} log_probs shape: {log_probs.shape}') # [seq_len, vocab_size]
    # logger.info(ranges)
    # logger.info(mut)
    score = 0.0
    for mut_str in mut:
        wt_res = mut_str[0]
        chain_id = mut_str[1]
        res_id = int(mut_str[2:-1]) - 1 # convert to 0-based index
        mut_res = mut_str[-1]
        if chain_id not in ranges:
            # logger.warning(f'Chain {chain_id} not found in {fpath} and chain_ids {chain_ids}, skipping mutation {mut_str}')
            continue
        start, end = ranges[chain_id]
        res_pos = start + res_id
        if res_pos < start or res_pos >= end:
            logger.warning(f'Residue position {res_id} out of range for chain {chain_id} in {fpath}, skipping mutation {mut_str}')
            continue
        wt_token_id = alphabet.get_idx(wt_res)
        mut_token_id = alphabet.get_idx(mut_res)
        wt_log_prob = log_probs[res_pos, wt_token_id].item()
        mut_log_prob = log_probs[res_pos, mut_token_id].item()
        score += mut_log_prob - wt_log_prob # higher means more stable

    return score

def stab_ddg_score(model, alphabet, fpath, chain_ids, mut, logger):
    bound_score = naive_score(model, alphabet, fpath, chain_ids, mut, logger)
    unbound_score_list = []
    for chain_id in chain_ids:
        unbound_score = naive_score(model, alphabet, fpath, [chain_id], mut, logger)
        unbound_score_list.append(unbound_score)
    final_score = bound_score - sum(unbound_score_list)

    return final_score

def get_args():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of ESM-IF')
    parser.add_argument('--logdir', type=str, default='logs/debug', help='Directory to save logs and results')
    parser.add_argument('--input_csv', type=str, default='data/SKEMPI-processed/filtered_skempi_test.csv', help='Path to input CSV file containing mutation data')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file for results')
    parser.add_argument('--pdb_dir', type=str, default='data/SKEMPI-v2/PDBs', help='Directory containing PDB files')
    parser.add_argument('--model_name', type=str, default='esm_if1_gvp4_t16_142M_UR50', help='ESM-IF model name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats to average over for each mutation')
    parser.add_argument('--no_timestamp', action='store_true', help='Whether to omit timestamp from log directory name')
    parser.add_argument('--tag', type=str, default='', help='Additional tag to include in log directory name')
    parser.add_argument('--sep', type=str, default=',', help='Separator used in the input CSV file')
    parser.add_argument('--score_method', type=str, default='naive', choices=['naive', 'stab_ddg'], help='Scoring method to use for evaluating mutations')
    return parser.parse_args()

def main():
    start_overall = time.time()
    args = get_args()
    commons.seed_all(args.seed)

    log_dir = commons.get_new_log_dir(args.logdir, prefix='zero_shot', tag=args.tag, timestamp=not args.no_timestamp)
    logger = commons.get_logger('ZeroShotESMIF', log_dir)
    logger.info(f'args:\n{args}')

    # Load ESM-IF model
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(args.model_name)
    model = model.eval()
    model.to(args.device)
    logger.info(f'Trainable parameters: {commons.count_parameters(model):,}')


    # load mutation data
    df = pd.read_csv(args.input_csv, sep=args.sep)
    # df = df.head(100) # for debugging
    pdb_mut_ddg = list(zip(df['#Pdb'], df['Mutation(s)_cleaned'], df['ddG']))
    pdb_chain_mut_ddg = []
    for i in range(len(pdb_mut_ddg)):
        pdb, mut, ddg = pdb_mut_ddg[i]
        pdb_id = pdb.strip().split('_')[0]
        chain_ids = parse_chain_ids(pdb)
        mut = mut.strip().split(',')
        ddg = float(ddg)
        pdb_chain_mut_ddg.append((pdb_id, chain_ids, mut, ddg))
    logger.info(f'Loaded {len(pdb_chain_mut_ddg)} PPI mutation-ddG data from {args.input_csv}')
    logger.info(f'Number of PPIs: {df["#Pdb"].nunique()}')
    assert len(pdb_chain_mut_ddg) == len(df), f'Expected {len(df)} entries but got {len(pdb_chain_mut_ddg)} after parsing'

    predictions = [np.nan] * len(pdb_chain_mut_ddg)
    for i, (pdb_id, chain_ids, mut, ddg) in enumerate(tqdm(pdb_chain_mut_ddg, desc='Processing mutations', dynamic_ncols=True)):
        fpath = os.path.join(args.pdb_dir, f'{pdb_id}.pdb')
        if not os.path.exists(fpath):
            logger.warning(f'PDB file not found: {fpath}, skipping {pdb_id}')
            continue
        
        score_list = []
        try:
            for repeat in range(args.num_repeats):
                if args.score_method == 'naive':
                    score_ = naive_score(model, alphabet, fpath, chain_ids, mut, logger)
                elif args.score_method == 'stab_ddg':
                    score_ = stab_ddg_score(model, alphabet, fpath, chain_ids, mut, logger)
                else:
                    raise ValueError(f"Unknown score method: {args.score_method}")
                score_list.append(score_)
            score = np.mean(score_list)
            # logger.info(f'{pdb_id} mutation score: {score:.4f}, ddG: {ddg:.4f}')
            predictions[i] = score
            # input()
        except Exception as e:
            logger.error(f'Error processing {pdb_id}: {e}')
            continue
    

    df['predicted_score'] = predictions

    # group by #Pdb, compute the spearman correlation between predicted_score and ddG inside each group, and log the results
    grouped = df.groupby('#Pdb')
    correlation_list = []
    count_threshold = 10
    for pdb, group in grouped:
        if len(group) > count_threshold:
            corr, _ = spearmanr(group['predicted_score'], group['ddG'])
            # logger.info(f'Spearman correlation for {pdb}: {corr:.4f}')
            correlation_list.append((pdb, corr))
        else:
            logger.info(f'Not enough data to compute Spearman correlation for {pdb}')
    
    logger.info(f'Computed Spearman correlation for {len(correlation_list)} / {len(grouped)} PPIs')
    correlation_df = pd.DataFrame(correlation_list, columns=['#Pdb', 'Spearmanr'])
    correlation_df.to_csv(os.path.join(log_dir, 'spearman_correlation.csv'), index=False, sep=args.sep)
    mean_sparman = correlation_df['Spearmanr'].mean()
    logger.info(f'Mean Spearman correlation across {len(correlation_list)} / {len(grouped)} PPIs: {mean_sparman:.4f}')

    args.output_csv = args.output_csv or os.path.join(log_dir, 'zero_shot_esmif_predictions.csv')
    df.to_csv(args.output_csv, index=False, sep=args.sep)

if __name__ == '__main__':
    main()