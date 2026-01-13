import sys
sys.path.append('/work/jiaqi/ppi-mut')
import esm.inverse_folding
import os
from tqdm import tqdm

data_dir = 'data/SKEMPI-v2/PDBs'
pdb_list = []
for f in os.listdir(data_dir):
    if f.endswith('.pdb') or f.endswith('.cif'):
        pdb_list.append(os.path.join(data_dir, f))
print(f'Found {len(pdb_list)} structures.')
fasta_path = 'data/SKEMPI-v2/complex_seqs.fasta'

for pdb in tqdm(pdb_list):
    try:
        structure = esm.inverse_folding.util.load_structure(pdb)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        pdb_id = os.path.basename(pdb).split('.')[0]
        with open(fasta_path, 'a+') as f:
            for chain_id, seq in native_seqs.items():
                f.write(f'>{pdb_id}_{chain_id}\n')
                f.write(f'{seq}\n')
    except Exception as e:
        print(f'Error processing {pdb}: {e}')