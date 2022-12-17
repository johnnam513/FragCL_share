
import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit.Chem.AllChem import ReactionFromSmarts


allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
        
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        #atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        #               [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature  = bond_to_feature_vector(bond)
            #edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
            #               [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    #conformer = mol.GetConformers()[0]
    #positions = conformer.GetPositions()
    #positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=None)
    return data



def mol_fragment(mol):
    if mol is None:
        print('something wrong')
    
    Rxn = ReactionFromSmarts('[*:1]-[*:2]>>[*:1].[*:2]')
    fragments = Rxn.RunReactants([mol])
    reactions = []
    for (f1, f2) in fragments:
        frag1 = Chem.MolToSmiles(f1)
        frag2 = Chem.MolToSmiles(f2)
        if set([frag1, frag2]) not in reactions:
            reactions.append(set([frag1, frag2]))
    
    min_frag_size_diff = -1
    balanced_rxn = None
    #print(reactions)
    for rxn_set in reactions:
        rxn_list = list(rxn_set)
        if len(rxn_list) != 2:
            continue
        if abs(len(rxn_list[0]) - len(rxn_list[1])) < min_frag_size_diff or min_frag_size_diff < 0:
            if Chem.MolFromSmiles(rxn_list[0]) is None or Chem.MolFromSmiles(rxn_list[1]) is None:
                #print(rxn_list[0])
                #print(rxn_list[1])
                continue
            balanced_rxn = rxn_list
            min_frag_size_diff = abs(len(rxn_list[0]) - len(rxn_list[1]))
    if balanced_rxn is None:
        #print("balanced_rxn is none")
        print(Chem.MolToSmiles(mol))
        print(reactions)
        return None
    if balanced_rxn is not None:
        if balanced_rxn[0].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
            #print("only C fragment")
            return None
        elif balanced_rxn[1].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
            #print("only C fragment")
            return None
        
    mol1 = Chem.MolFromSmiles(balanced_rxn[0])
    mol2 = Chem.MolFromSmiles(balanced_rxn[1])

    return mol1, mol2

def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '{}/rdkit_folder'.format(data_folder)
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        ##### Path should match #####
        if sub_dic.get('pickle_path', '') == '':
            continue

        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            # conf['boltzmannweight'] a float for the conformer (a few rotamers)
            # conf['conformerweights'] a list of fine weights of each rotamer
            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class Molecule3DDataset(InMemoryDataset):

    def __init__(self, root, n_mol, n_conf, n_upper, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)
        if 'smiles_copy_from_3D_file' in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs['smiles_copy_from_3D_file']
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(Molecule3DDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data_smiles_list = []
        data1_list = []
        data2_list = []

        downstream_task_list = ["tox21", "toxcast", "clintox", "bbbp", "sider", "muv", "hiv", "bace", "esol", "lipophilicity"]
        
        '''
        whole_SMILES_set = set()
        for task in downstream_task_list:
            print("====== {} ======".format(task))
            file_path = "../datasets/molecule_datasets/{}/processed/smiles.csv".format(task)
            SMILES_list = load_SMILES_list(file_path)
            temp_SMILES_set = set(SMILES_list)
            whole_SMILES_set = whole_SMILES_set | temp_SMILES_set
        print("len of downstream SMILES:", len(whole_SMILES_set))

        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            dir_name = '{}/rdkit_folder'.format(data_folder)
            drugs_file = '{}/summary_drugs.json'.format(dir_name)
            with open(drugs_file, 'r') as f:
                drugs_summary = json.load(f)
            drugs_summary = list(drugs_summary.items())
            print('# of SMILES: {}'.format(len(drugs_summary)))
            # expected: 304,466 molecules

            random.seed(self.seed)
            random.shuffle(drugs_summary)
            mol_idx, idx, notfound = 0, 0, 0
            for smiles, sub_dic in tqdm(drugs_summary):
                if smiles in whole_SMILES_set:
                    continue
                ##### Path should match #####
                if sub_dic.get('pickle_path', '') == '':
                    notfound += 1
                    continue

                mol_path = join(dir_name, sub_dic['pickle_path'])
                with open(mol_path, 'rb') as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic['conformers']

                    ##### count should match #####
                    conf_n = len(conformer_list)
                    if conf_n < self.n_conf or conf_n > self.n_upper:
                        # print(smiles, len(conformer_list))
                        notfound += 1
                        continue

                    ##### SMILES should match #####
                    #  export prefix=https://github.com/learningmatter-mit/geom
                    #  Ref: ${prefix}/issues/4#issuecomment-853486681
                    #  Ref: ${prefix}/blob/master/tutorials/02_loading_rdkit_mols.ipynb
                    conf_list = [
                        Chem.MolToSmiles(
                            Chem.MolFromSmiles(
                                Chem.MolToSmiles(rd_mol['rd_mol'])))
                        for rd_mol in conformer_list[:self.n_conf]]

                    conf_list_raw = [
                        Chem.MolToSmiles(rd_mol['rd_mol'])
                        for rd_mol in conformer_list[:self.n_conf]]
                    # check that they're all the same
                    same_confs = len(list(set(conf_list))) == 1
                    same_confs_raw = len(list(set(conf_list_raw))) == 1
                    if not same_confs:
                        if same_confs_raw is True:
                            print("Interesting")
                        notfound += 1
                        continue

                    for conformer_dict in conformer_list[:self.n_conf]:
                        # select the first n_conf conformations
                        rdkit_mol = conformer_dict['rd_mol']
                        data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                        data.id = torch.tensor([idx])
                        data.mol_id = torch.tensor([mol_idx])
                        data_smiles_list.append(smiles)
                        data_list.append(data)
                        idx += 1
                        # print(data.id, '\t', data.mol_id)

                # select the first n_mol molecules
                if mol_idx + 1 >= self.n_mol:
                    break
                if same_confs:
                    mol_idx += 1

            print('mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}'.format(
                mol_idx, len(data_smiles_list), len(set(data_smiles_list))))
        '''
        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            print('something wrong')
        else:  # 2D datasets
            with open(self.smiles_copy_from_3D_file, 'r') as f:
                lines = f.readlines()
            for smiles in lines:
                data_smiles_list.append(smiles.strip())
            data_smiles_list = list(dict.fromkeys(data_smiles_list))

            # load 3D structure
            dir_name = '{}/rdkit_folder'.format(data_folder)
            drugs_file = '{}/summary_drugs.json'.format(dir_name)
            with open(drugs_file, 'r') as f:
                drugs_summary = json.load(f)
            # expected: 304,466 molecules
            print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

            mol_idx, idx, notfound = 0, 0, 0
            err_cnt = 0

            for smiles in tqdm(data_smiles_list):
                sub_dic = drugs_summary[smiles]
                mol_path = join(dir_name, sub_dic['pickle_path'])
                with open(mol_path, 'rb') as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic['conformers']
                    conformer = conformer_list[0]
                    rdkit_mol = conformer['rd_mol']
                    mols = mol_fragment(rdkit_mol)


                    #data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    data = mol_to_graph_data(rdkit_mol)

                    if mols is None:
                        err_cnt += 1
                        continue
                    else:
                        data1 = mol_to_graph_data(mols[0])
                        data2 = mol_to_graph_data(mols[1])

                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    data1.mol_id = torch.tensor([mol_idx])
                    data1.id = torch.tensor([idx])
                    data1_list.append(data1)
                    data2.mol_id = torch.tensor([mol_idx])
                    data2.id = torch.tensor([idx])
                    data2_list.append(data2)
                    mol_idx += 1
                    idx += 1


                    
                    if len(data1.edge_attr) == 0:
                        data1.edge_attr = torch.empty(size=[0,3])
                    if len(data2.edge_attr) == 0:
                        data2.edge_attr = torch.empty(size=[0,3])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            data1_list = [data1 for data1 in data1_list if self.pre_filter(data1)]
            data2_list = [data2 for data2 in data2_list if self.pre_filter(data2)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            data1_list = [self.pre_transform(data1) for data1 in data1_list]
            data2_list = [self.pre_transform(data2) for data2 in data2_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)
        
        data, slices = self.collate(data_list)
        data1, slices1 = self.collate(data1_list)
        data2, slices2 = self.collate(data2_list)
        
        torch.save((data, slices), self.processed_paths[0])
        torch.save((data1, slices1), self.processed_paths[0] + "_1")
        torch.save((data2, slices2), self.processed_paths[0] + "_2")

        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
        return


def load_SMILES_list(file_path):
    SMILES_list = []
    with open(file_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            SMILES_list.append(line.strip().decode())

    return SMILES_list


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, help='number of unique smiles/molecules')
    parser.add_argument('--n_conf', type=int, help='number of conformers of each molecule')
    parser.add_argument('--n_upper', type=int, help='upper bound for number of conformers')
    parser.add_argument('--data_folder', type=str)
    args = parser.parse_args()

    data_folder = args.data_folder

    if args.sum:
        sum_list = summarise()
        with open('{}/summarise.json'.format(data_folder), 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol, n_conf, n_upper = args.n_mol, args.n_conf, args.n_upper
        root_2d = '{}/GEOM_2D_nmol{}_any_morefeat'.format(data_folder, n_mol, n_conf, n_upper)
        root_3d = '{}/GEOM_3D_nmol{}_nconf{}_nupper{}_morefeat'.format(data_folder, n_mol, n_conf, n_upper)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        #Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)
        # Generate 2D Datasets (2D SMILES)
        Molecule3DDataset(root=root_2d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper,
                          smiles_copy_from_3D_file='/home/osikjs/GraphMVP/datasets/GEOM_2D_nmol50000_bug/processed/smiles.csv')
    
    ##### to data copy to SLURM_TMPDIR under the `datasets` folder #####
    '''
    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    mv 4327252 rdkit_folder.tar.gz
    cp rdkit_folder.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xvf rdkit_folder.tar.gz
    '''

    ##### for data pre-processing #####
    '''
    python GEOM_dataset_preparation.py --n_mol 100 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
    python GEOM_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
    '''