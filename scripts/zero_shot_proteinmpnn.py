import sys
sys.path.append('/work/jiaqi/ppi-mut')

from stabddg.ppi_dataset import SKEMPIDataset
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG

import torch
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import argparse, os, json, time, pickle
import numpy as np

from utils import commons

def get_args():
    parser = argparse.ArgumentParser(description='Zero-shot prediction of ddG using ProteinMPNN')
    parser.add_argument('--logdir', type=str, default='logs/debug')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--tag', type=str, default='', help='Additional tag to include in log directory name')
    parser.add_argument('--no_timestamp', action='store_true', help='Whether to omit timestamp from log directory name')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats for each mutation to average predictions')
    parser.add_argument('--csv_path', type=str, default='data/SKEMPI-processed/filtered_skempi_test.csv', help='Path to the CSV file containing the dataset')
    parser.add_argument('--pdb_dir', type=str, default='data/SKEMPI-v2/PDBs', help='Directory containing the PDB files')
    parser.add_argument('--pdb_dict_cache_path', type=str, default='data/SKEMPI-processed/filtered_skempi_test.pkl', help='Path to cache the processed PDB structures')
    parser.add_argument('--ckpt_path', type=str, default='/work/jiaqi/ProteinMPNN/vanilla_model_weights/v_48_010.pt', help='Path to the ProteinMPNN checkpoint')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level for StaBddG model')
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of tokens to process in a batch (not number of sequences, will be divided by sequence length to get number of sequences per batch)")
    parser.add_argument('--use_naive', action='store_true', help='Whether to use the naive scoring method instead of the antithetic variates method in StaBddG')

    return parser.parse_args()

def main():
    start_overall = time.time()
    args = get_args()
    commons.seed_all(args.seed)

    log_dir = commons.get_new_log_dir(args.logdir, prefix='zero_shot_pmpnn', tag=args.tag, timestamp=not args.no_timestamp)
    logger = commons.get_logger('ZeroShotMPNN', log_dir)
    logger.info(f'args:\n{args}')

    dataset = SKEMPIDataset(csv_path=args.csv_path,
                            pdb_dir=args.pdb_dir,
                            pdb_dict_cache_path=args.pdb_dict_cache_path
                            )
    logger.info(f'Loaded dataset with {len(dataset)} complexes from {args.csv_path}')

    pmpnn = ProteinMPNN(node_features=128, 
                        edge_features=128, 
                        hidden_dim=128,
                        num_encoder_layers=3, 
                        num_decoder_layers=3, 
                        k_neighbors=48, 
                        dropout=0.0,
                        augment_eps=0.0)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint.keys():
        pmpnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        pmpnn.load_state_dict(checkpoint)
    logger.info(f'Successfully loaded ProteinMPNN model at {args.ckpt_path}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = StaBddG(pmpnn=pmpnn, noise_level=args.noise_level, device=device)
    model.to(device)
    model.eval()
    logger.info(f'Model: {model}')
    logger.info(f'Trainable parameters: {commons.count_parameters(model):,}')

    pred_df = []
    val_spearman = []
    val_pearson = []
    val_rmse = []

    for sample in tqdm(dataset, desc="Interface", dynamic_ncols=True):  
        complex, binder1, binder2 = sample['complex'], sample['binder1'], sample['binder2']
        complex_mut_seqs = sample['complex_mut_seqs'].to(device)
        binder1_mut_seqs = sample['binder1_mut_seqs'].to(device)
        binder2_mut_seqs = sample['binder2_mut_seqs'].to(device)
        ddG = sample['ddG']

        binding_ddG_pred_ensemble = []
        for _ in range(args.num_repeats):
            with torch.no_grad(): 
                N = complex_mut_seqs.shape[0]
                M = args.batch_size // complex_mut_seqs.shape[1] # convert number of tokens to number of sequences per batch

                binding_ddG_pred_ = []
                for batch_idx in range(0, N, M):
                    B = min(N - batch_idx, M)
                    batch_binding_ddG_pred = model(complex, binder1, binder2, complex_mut_seqs[batch_idx:batch_idx+B], 
                                                binder1_mut_seqs[batch_idx:batch_idx+B], binder2_mut_seqs[batch_idx:batch_idx+B], use_naive=args.use_naive)
                    binding_ddG_pred_.append(batch_binding_ddG_pred.cpu().detach())

                binding_ddG_pred_ = torch.cat(binding_ddG_pred_)
                
                binding_ddG_pred_ensemble.append(binding_ddG_pred_.squeeze().cpu())
                
        binding_ddG_pred = torch.stack(binding_ddG_pred_ensemble).mean(dim=0)

        name, mutations = sample['name'], sample['mutation_list']
        data = {
            "#Pdb": [name] * len(mutations),  # Repeat the name for all rows
            "Mutation": mutations,
            "ddG": ddG.cpu().detach().numpy(),
            "Prediction": binding_ddG_pred.cpu().detach().numpy()
        }

        df = pd.DataFrame(data)
        pred_df.append(df)

        sp, _ = spearmanr(binding_ddG_pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())

        pr, _ = pearsonr(binding_ddG_pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())

        rmse = torch.sqrt(torch.mean((binding_ddG_pred.cpu() - ddG.cpu()) ** 2)).item()

        val_spearman.append(sp)
        val_pearson.append(pr)
        val_rmse.append(rmse)

    pred_df = pd.concat(pred_df, ignore_index=True)
    pred_df.to_csv(os.path.join(log_dir, 'predictions.csv'), index=False)
    logger.info(f'Mean Spearman correlation: {np.mean(val_spearman):.4f} ± {np.std(val_spearman):.4f}')

    grouped = pred_df.groupby('#Pdb')
    correlation_list = []
    count_threshold = 10
    for pdb, group in grouped:
        if len(group) >= count_threshold:
            corr, _ = spearmanr(group['Prediction'], group['ddG'])
            # logger.info(f'Spearman correlation for {pdb}: {corr:.4f}')
            correlation_list.append((pdb, corr))
        else:
            logger.info(f'{pdb}: {len(group)} < {count_threshold}, not enough data to compute Spearman correlation')
    
    logger.info(f'Computed Spearman correlation for {len(correlation_list)} / {len(grouped)} PPIs')
    correlation_df = pd.DataFrame(correlation_list, columns=['#Pdb', 'Spearmanr'])
    correlation_df.to_csv(os.path.join(log_dir, 'spearman_correlation.csv'), index=False)
    mean_spearman = correlation_df['Spearmanr'].mean()
    std_spearman = correlation_df['Spearmanr'].std()
    logger.info(f'Mean Spearman correlation across {len(correlation_list)} / {len(grouped)} PPIs: {mean_spearman:.4f} ± {std_spearman:.4f}')

if __name__ == '__main__':
    main()