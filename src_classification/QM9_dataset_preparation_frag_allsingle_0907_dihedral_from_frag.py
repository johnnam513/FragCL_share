import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join
import copy
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasets import allowable_features
from rdkit.Chem.AllChem import ReactionFromSmarts
from scipy.sparse.csgraph import connected_components
import math
from atom3d.datasets import LMDBDataset
from atom3d.util.transforms import mol_graph_transform

class GNNTransformSMP(object):
    def __init__(self, label_name):
        self.label_name = label_name

    def _lookup_label(self, item, name):
        if 'label_mapping' not in self.__dict__:
            label_mapping = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
                ]
            self.label_mapping = {k: v for v, k in enumerate(label_mapping)}
        return (item['labels'][self.label_mapping[name]] ) 

    def __call__(self, item):
        item = mol_graph_transform(item, 'atoms', 'labels', allowable_atoms=['C', 'H', 'O', 'N', 'F'], use_bonds=True, onehot_edges=True)
        graph = item['atoms']
        #x2 = torch.tensor(item['atom_feats'], dtype=torch.float).t().contiguous()
        #graph.x = torch.cat([graph.x.to(torch.float), x2], dim=-1)
        graph.y = self._lookup_label(item, self.label_name)
        graph.id = item['id']
        return graph

def mol_fragment(mol):
    if mol is None:
        print('something wrong')
    
    Rxn = ReactionFromSmarts('[*:1]-!@[*:2]>>[*:1].[*:2]')
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
    #if balanced_rxn is not None:
    #    if balanced_rxn[0].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
    #        #print("only C fragment")
    #        return None
    #    elif balanced_rxn[1].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
            #print("only C fragment")
    #        return None
        
    mol1 = Chem.MolFromSmiles(balanced_rxn[0])
    mol2 = Chem.MolFromSmiles(balanced_rxn[1])

    return mol1, mol2



def mol_combination(mol1, mol2):
    Rxn = ReactionFromSmarts('[*;!H0:1].[*;!H0:2]>>[*:1]-[*:2]')
    combination = Rxn.RunReactants([mol1, mol2])
    if combination is None:
        raise 'combination error'
    else:
        return combination[0][0]
    


def mol_to_graph_data_obj_simple_3D_gen(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    mol = Chem.AddHs(mol)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
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

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    #Allchem.MMFFOptimizeMolceule(mol)
    
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data




def check_connected(edge_list):
    connected_comps = []
    for (i,j) in edge_list:
        done=False

        for comps in connected_comps:
            if i in comps and j in comps:
                done=True
                break
            elif i in comps:
                done=True
                done2=False
                for (k,comps2) in enumerate(connected_comps):
                    if j in comps2:
                        comps = comps + comps2
                        connected_comps = connected_comps[:k] + connected_comps[k+1:]
                        done2=True
                if done2==False:
                    comps.append(j)
                break
            elif j in comps:
                done=True
                done2=False
                for (k,comps2) in enumerate(connected_comps):
                    if i in comps2:
                        comps = comps + comps2
                        connected_comps = connected_comps[:k] + connected_comps[k+1:]
                        done2=True
                if done2==False:
                    comps.append(i)
                break

        if done == False:
            connected_comps.append([i,j])
    print(connected_comps)
    print("\n")


def mol_to_graph_data_obj_simple_3D(graph):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    #print(Chem.MolToSmiles(mol))
    #atom_features_list = []
    #for atom in mol.GetAtoms():
    #    atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
    #                   [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
    #    atom_features_list.append(atom_feature)
    #x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    
    
    # bonds, two features: bond type, bond direction
    if len(graph.x) > 0:  # mol has bonds
    #    edges_list = []
    #    edge_features_list = []
    #    single_bond_list = []
    #    for bond in mol.GetBonds():
    #        i = bond.GetBeginAtomIdx()
    #        j = bond.GetEndAtomIdx()
    #        if allowable_features['possible_bonds'].index(bond.GetBondType()) == 0:
    #            single_bond_list.append((i,j))
    #        edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
    #                       [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
    #        edges_list.append((i, j))
    #        edge_features_list.append(edge_feature)
    #        edges_list.append((j, i))
    #        edge_features_list.append(edge_feature)
    #   
        
        x = graph.x.argmax(dim=1).unsqueeze(1)
        
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
        
        edge_attr = graph.edge_attr.argmax(dim=1).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, torch.zeros_like(edge_attr)], dim=1)
        edge_index = graph.edge_index

        #print(x.shape)
        #print(edge_attr.shape)
        #print(edge_index.shape)
        #print('a')
        #print(x.shape)
        #print('b')
        #print(edge_attr.shape)
        #print('c')
        #print(edge_index.shape)
        
        
        single_bond_list = []
        edges_list = []
        edge_features_list = []
        for (i,edge) in enumerate(edge_index.T):
            tmp_edge = list(edge)
            
            edges_list.append((tmp_edge[0].item(), tmp_edge[1].item()))
            
            if edge_attr[i][0] == 0:
                single_bond_list.append((tmp_edge[0].item(), tmp_edge[1].item()))
        for (i,edge) in enumerate(edge_attr):
            tmp_edge = list(edge)
            edge_features_list.append((tmp_edge[0].item(), tmp_edge[1].item()))
          
        atom_features_list = []
        for (i,atom) in enumerate(x):
            tmp_atom = list(atom)
            atom_features_list.append((tmp_atom[0].item(), tmp_atom[1].item()))
        num_atoms = len(x)
        
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
        
        anchor1 = None
        anchor2 = None
        
        for (i,j) in edges_list:
            #print(i,j)
            if x[i][0].item() == 1 or x[j][0].item() == 1:
                continue
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
        if anchor1 == None or anchor2 == None:
            return []
        
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
    #conformer = mol.GetConformers()[0]
    #positions = conformer.GetPositions()
    
    positions = graph.pos
    
    # anchor1_position0 = torch.Tensor(positions[anchor1[0]])
    # anchor1_position1 = torch.Tensor(positions[anchor1[1]])
    # anchor2_position0 = torch.Tensor(positions[anchor2[0]])
    # anchor2_position1 = torch.Tensor(positions[anchor2[1]])
    
    # anchor1_angle = math.acos(torch.inner(anchor1_position1 - anchor1_position0, anchor1_position1 - anchor2_position0) /(torch.norm(anchor1_position1 - anchor1_position0) * torch.norm(anchor1_position1 - anchor2_position0)) )
    # anchor2_angle = math.acos(torch.inner(anchor2_position0 - anchor1_position1, anchor2_position0 - anchor2_position1) /(torch.norm(anchor2_position0 - anchor1_position1) * torch.norm(anchor2_position0 - anchor2_position1)) )
    
    
    # anchor1_ang = torch.Tensor([[frag1_dict[anchor1[0]], frag1_dict[anchor1[1]], frag2_dict[anchor2[0]]]]).long()
    # anchor2_ang = torch.Tensor([[frag1_dict[anchor1[1]], frag2_dict[anchor2[0]], frag2_dict[anchor2[1]]]]).long()
    # anchor1_angle = int(anchor1_angle / np.pi * 9)
    # anchor2_angle = int(anchor2_angle / np.pi * 9)
    
    
    
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
    dihedral_angle_value = np.degrees(np.arctan2(y, x10))

    positions1 = []
    positions2 = []
    
    for i in range(len(positions)):
        
        if frag_adj1[i] == True:
            positions1.append(positions[i].unsqueeze(0))
        else:
            positions2.append(positions[i].unsqueeze(0))
    
    positions = torch.Tensor(positions)
    
    positions1 = torch.cat(positions1, dim=0)
    positions2 = torch.cat(positions2, dim=0)
    
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions,  dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle, dihedral_angle_value=dihedral_angle_value)
    data1 = Data(x=x1, edge_index=edge_index1,
                edge_attr=edge_attr1, positions=positions1, dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle, dihedral_angle_value=dihedral_angle_value)
    data2 = Data(x=x2, edge_index=edge_index2,
                edge_attr=edge_attr2, positions=positions2, dihedral_anchor=dihedral_ang, dihedral_angle=dihedral_angle, dihedral_angle_value=dihedral_angle_value)
    
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
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
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

    def __init__(self, root, n_mol, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)
        if 'smiles_copy_from_3D_file' in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs['smiles_copy_from_3D_file']
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol = n_mol
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(Molecule3DDatasetFrag, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data1, self.slices1 = torch.load(self.processed_paths[0] + "_1")
            self.data2, self.slices2 = torch.load(self.processed_paths[0] + "_2")
        print(self.data)
        print(self.data1)
        print(self.data2)
        
        print('root: {},\ndata: {},\nn_mol: {},\n'.format(
            self.root, self.data, self.n_mol))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]

            item1, slices1 = self.data1[key], self.slices1[key]
            s = list(repeat(slice(None), item.dim()))
            s[data1.__cat_dim__(key, item1)] = slice(slices1[idx], slices1[idx+1])
            data1[key] = item1[s]

            item2, slices2 = self.data2[key], self.slices2[key]
            s = list(repeat(slice(None), item.dim()))
            s[data2.__cat_dim__(key, item2)] = slice(slices2[idx], slices2[idx+1])
            data2[key] = item2[s]
        return data, data1, data2

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

        downstream_task_list = ["tox21", "toxcast", "clintox", "bbbp", "sider", "muv", "hiv", "bace", "esol", "lipophilicity"]
        whole_SMILES_set = set()
        '''
        for task in downstream_task_list:
            print("====== {} ======".format(task))
            file_path = "../datasets/molecule_datasets/{}/processed/smiles.csv".format(task)
            SMILES_list = load_SMILES_list(file_path)
            temp_SMILES_set = set(SMILES_list)
            whole_SMILES_set = whole_SMILES_set | temp_SMILES_set
        print("len of downstream SMILES:", len(whole_SMILES_set))
        '''
        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            print('something wrong')
        else:  # 2D datasets
            #with open(self.smiles_copy_from_3D_file, 'r') as f:
            #    lines = f.readlines()
            #for smiles in lines:
            #    data_smiles_list.append(smiles.strip())
            #data_smiles_list = list(dict.fromkeys(data_smiles_list))

            # load 3D structure
            #dir_name = '{}/rdkit_folder'.format(data_folder)
            #drugs_file = '{}/summary_drugs.json'.format(dir_name)
            #with open(drugs_file, 'r') as f:
            #    drugs_summary = json.load(f)
            # expected: 304,466 molecules
            #print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

            mol_idx, idx, notfound = 0, 0, 0
            err_cnt = 0
            one_functional_group_list = []
            train_dataset = LMDBDataset('/home/osikjs/atom3d/data/random/data/train', transform=GNNTransformSMP("mu"))
            for qm9_data in tqdm(train_dataset):
                
                #sub_dic = drugs_summary[smiles]
                #mol_path = join(dir_name, sub_dic['pickle_path'])
                #with open(mol_path, 'rb') as f:
                    #mol_dic = pickle.load(f)
                    #conformer_list = mol_dic['conformers']
                    #new_list = sorted(conformer_list, key=lambda d: d['relativeenergy']) 


                    #conformer = new_list[0]
                    #rdkit_mol = conformer['rd_mol']
                    
                    #data = mol_to_graph_data(rdkit_mol)
                try:
                    data, data1, data2 = mol_to_graph_data_obj_simple_3D(qm9_data)
                    #print('done')
                except:
                    continue
                data.mol_id = torch.tensor([mol_idx])
                data.id = torch.tensor([idx])
                #data.fp = fp
                data_list.append(data)
                data1.mol_id = torch.tensor([mol_idx])
                data1.id = torch.tensor([idx])
                data1_list.append(data1)
                data2.mol_id = torch.tensor([mol_idx])
                data2.id = torch.tensor([idx])
                data2_list.append(data2)

                mol_idx += 1
                idx += 1
                

        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]
    
        #data_smiles_series = pd.Series(data_smiles_list)
        #saver_path = join(self.processed_dir, 'smiles.csv')
        #print('saving to {}'.format(saver_path))
        #data_smiles_series.to_csv(saver_path, index=False, header=False)
    
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
    parser.add_argument('--data_folder', type=str)
    args = parser.parse_args()

    data_folder = args.data_folder

    if args.sum:
        sum_list = summarise()
        with open('{}/summarise.json'.format(data_folder), 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol = args.n_mol
        root_2d = '{}/QM9_2D_3D_cut_singlebond_manual_alin16'.format(data_folder)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        #Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)
        # Generate 2D Datasets (2D SMILES)
        Molecule3DDatasetFrag(root=root_2d, n_mol=n_mol,
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
