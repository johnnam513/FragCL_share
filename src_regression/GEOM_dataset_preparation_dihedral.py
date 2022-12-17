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
from scipy.sparse.csgraph import connected_components
import math
import copy

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


allowable_features = {
    'possible_atomic_num_list':       list(range(0, 119)),
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
        single_bond_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if allowable_features['possible_bonds'].index(bond.GetBondType()) == 0:
                single_bond_list.append((i,j))
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        #edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        #edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)


        num_atoms = len(mol.GetAtoms())
        
        frag_adj1 = []
        frag_adj2 = []
        
        final_list = None
        min_diff = 1000
        for (i,j) in single_bond_list:
            adj = np.zeros((num_atoms, num_atoms))
            tmp_list = copy.deepcopy(edges_list)
            tmp_list.remove((i,j))
            tmp_list.remove((j,i))
            for (i1, j1) in tmp_list:
                adj[i1,j1] = 1
            num_components, comp_indices = connected_components(adj)
            #print(comp_indices)
            if num_components == 2:
                num_comp1 = (comp_indices==0).sum()
                num_comp2 = (comp_indices==1).sum()
                if abs(num_comp1-num_comp2) < min_diff:
                    frag_adj1 = (comp_indices == 0)
                    frag_adj2 = (comp_indices == 1)
                    min_diff = abs(num_comp1-num_comp2)
                    final_list = tmp_list
                    final_num_comp1 = num_comp1
                    final_num_comp2 = num_comp2
        
        #print(final_num_comp1)
        #print(final_num_comp2)
        #if final_num_comp1 < 4:
        #    return None
        #elif final_num_comp2 < 4:
        #    return None

        if len(frag_adj1) == 0:
            return None
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        
        frag1_dict = {}
        frag2_dict = {}

        idx1 = 0
        idx2 = 0
        for i, val in enumerate(frag_adj1):
            if val == True:
                frag1_dict[i] = idx1
                idx1 += 1
            if val == False:
                frag2_dict[i] = idx2
                idx2 += 1

        edges_list1 = []
        edges_list2 = []
        
        for (i,j) in edges_list:
            if (i,j) not in final_list:
                missing_edges = (i,j)
                continue
            if frag_adj1[i] == True and frag_adj1[j] == True:
                edges_list1.append((frag1_dict[i],frag1_dict[j]))
            elif frag_adj1[i] == False and frag_adj1[j] == False:
                edges_list2.append((frag2_dict[i],frag2_dict[j]))
            else:
                print('>>>>???????')

        for (i,j) in edges_list:
            #print(i,j)
            if (i,j) == missing_edges:
                continue
            if (j,i) == missing_edges:
                continue
            if i == missing_edges[0] and i in list(frag2_dict.keys()):
                anchor2 = (i,j)
            if i == missing_edges[0] and i in list(frag1_dict.keys()):
                anchor1 = (j,i)
            if i == missing_edges[1] and i in list(frag1_dict.keys()):
                anchor1 = (j,i)
            if i == missing_edges[1] and i in list(frag2_dict.keys()):
                anchor2 = (i,j)

        #print(missing_edges)
        #print(anchor1)
        #print(anchor2)


        #print(frag1_dict)
        #print(frag2_dict)
        edge_index1 = torch.tensor(np.array(edges_list1).T, dtype=torch.long)
        edge_index2 = torch.tensor(np.array(edges_list2).T, dtype=torch.long)
        
        
        atom_features_list1 = []
        atom_features_list2 = []
        for i in range(len(atom_features_list)):
            if frag_adj1[i] == True:
                atom_features_list1.append(atom_features_list[i])
            else:
                atom_features_list2.append(atom_features_list[i])

        x1 = torch.tensor(np.array(atom_features_list1), dtype=torch.long)
        x2 = torch.tensor(np.array(atom_features_list2), dtype=torch.long)

        edge_features_list1 = []
        edge_features_list2 = []
        idx = 0
        for (i,j) in edges_list:
            if (i,j) not in final_list:
                idx += 1
                continue
            elif frag_adj1[i] == True and frag_adj1[j] == True:
                edge_features_list1.append(edge_features_list[idx])
                idx += 1
            elif frag_adj1[i] == False and frag_adj1[j] == False:
                edge_features_list2.append(edge_features_list[idx])
                idx += 1
            else:
                print('>>>>???????')
        #print(idx)
        #print(len(edge_features_list))
        
        

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        edge_attr1 = torch.tensor(np.array(edge_features_list1), dtype=torch.long)
        edge_attr2 = torch.tensor(np.array(edge_features_list2), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        return []

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    
    anchor1_position0 = torch.Tensor(positions[anchor1[0]])
    anchor1_position1 = torch.Tensor(positions[anchor1[1]])
    anchor2_position0 = torch.Tensor(positions[anchor2[0]])
    anchor2_position1 = torch.Tensor(positions[anchor2[1]])
    
    anchor1_angle = math.acos(torch.inner(anchor1_position1 - anchor1_position0, anchor1_position1 - anchor2_position0) /(torch.norm(anchor1_position1 - anchor1_position0) * torch.norm(anchor1_position1 - anchor2_position0)) )
    anchor2_angle = math.acos(torch.inner(anchor2_position0 - anchor1_position1, anchor2_position0 - anchor2_position1) /(torch.norm(anchor2_position0 - anchor1_position1) * torch.norm(anchor2_position0 - anchor2_position1)) )
    
    
    anchor1_ang = torch.Tensor([[frag1_dict[anchor1[0]], frag1_dict[anchor1[1]], frag2_dict[anchor2[0]]]]).long()
    anchor2_ang = torch.Tensor([[frag1_dict[anchor1[1]], frag2_dict[anchor2[0]], frag2_dict[anchor2[1]]]]).long()
    anchor1_angle = int(anchor1_angle / np.pi * 9)
    anchor2_angle = int(anchor2_angle / np.pi * 9)
    
    
    
    #dihedral_ang = np.array([anchor1[0], anchor1[1], anchor2[0], anchor2[1]])
    dihedral_ang = torch.Tensor([[frag1_dict[anchor1[0]], frag1_dict[anchor1[1]], frag2_dict[anchor2[0]], frag2_dict[anchor2[1]]]]).long()
    
    p0 = np.array(positions[anchor1[0]])
    p1 = np.array(positions[anchor1[1]])
    p2 = np.array(positions[anchor2[0]])
    p3 = np.array(positions[anchor2[1]])
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x10 = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dihedral_angle = int(np.degrees(np.arctan2(y, x10))/20) + 8
    

    positions1 = []
    positions2 = []
    for i in range(len(positions)):
        if frag_adj1[i] == True:
            positions1.append(positions[i])
        else:
            positions2.append(positions[i])
    positions = torch.Tensor(positions)
    
    positions1 = torch.Tensor(positions1)
    positions2 = torch.Tensor(positions2)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions, anchor1 = anchor1_ang, anchor1_angle = anchor1_angle, anchor2=anchor2_ang, anchor2_angle=anchor2_angle, dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle)
    data1 = Data(x=x1, edge_index=edge_index1,
                edge_attr=edge_attr1, positions=positions1, anchor1 = anchor1_ang, anchor1_angle = anchor1_angle, anchor2=anchor2_ang, anchor2_angle=anchor2_angle, dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle)
    data2 = Data(x=x2, edge_index=edge_index2,
                edge_attr=edge_attr2, positions=positions2, anchor1 = anchor1_ang, anchor1_angle = anchor1_angle, anchor2=anchor2_ang, anchor2_angle=anchor2_angle, dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle)
    #print('done?')
    #print(Chem.MolToSmiles(mol))
    #print(data['x'])
    #print(data1['x'])
    #print(data2['x'])
    
    #print(data['edge_index'])
    #print(data1['edge_index'])
    #print(data2['edge_index'])

    #print(data['edge_attr'])
    #print(data1['edge_attr'])
    #print(data2['edge_attr'])

    
    return data, data1, data2


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


class Molecule3DDatasetFrag(InMemoryDataset):

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

        super(Molecule3DDatasetFrag, self).__init__(
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
        data1_list = []
        data2_list = []
        data_smiles_list = []

        #downstream_task_list = ["tox21", "toxcast", "clintox", "bbbp", "sider", "muv", "hiv", "bace", "esol", "lipophilicity"]
        #whole_SMILES_set = set()
        #for task in downstream_task_list:
        #    print("====== {} ======".format(task))
        #    file_path = "../datasets/molecule_datasets/{}/processed/smiles.csv".format(task)
        #    SMILES_list = load_SMILES_list(file_path)
        #    temp_SMILES_set = set(SMILES_list)
        #    whole_SMILES_set = whole_SMILES_set | temp_SMILES_set
        #print("len of downstream SMILES:", len(whole_SMILES_set))

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

            for smiles in tqdm(data_smiles_list):
                sub_dic = drugs_summary[smiles]
                mol_path = join(dir_name, sub_dic['pickle_path'])
                with open(mol_path, 'rb') as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic['conformers']
                    conformer = conformer_list[0]
                    rdkit_mol = conformer['rd_mol']
                    
                    try:
                        
                        data, data1, data2 = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    #print('dd')
                    except:
                        continue
                    #data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    
                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    
                    data1.mol_id = torch.tensor([mol_idx])
                    data1.id = torch.tensor([idx])
                    data1_list.append(data1)

                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data2_list.append(data2)
                    
                    
                    mol_idx += 1
                    idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

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
        root_2d = '{}/GEOM_2D_nmol{}_nconf{}_nupper{}_morefeat'.format(data_folder, n_mol, n_conf, n_upper)
        root_3d = '{}/GEOM_3D_nmol{}_nconf{}_nupper{}_morefeat'.format(data_folder, n_mol, n_conf, n_upper)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        #Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)
        # Generate 2D Datasets (2D SMILES)
        Molecule3DDatasetFrag(root=root_2d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper,
                          smiles_copy_from_3D_file='../datasets/GEOM_2D_nmol50000_bug/processed/smiles.csv')
    
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