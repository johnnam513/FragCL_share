import argparse
from textwrap import dedent
import time

import numpy as np
import torch
import torch.optim as optim
from models import GNN
#from pretrain_JOAO import graphcl
from torch_geometric.data import DataLoader

from datasets import Molecule3DDatasetFragRandomaug
from os.path import join
import vision_transformer as vits
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        #self.pool = GlobalAttention(gate_nn = torch.nn.Linear(300, 1))
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300))
        self.aux = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1),
            nn.Sigmoid())

    def forward_cl_aux(self, x, edge_index, edge_attr, batch, positions):
        x = self.gnn(x, edge_index, edge_attr)
        gen_p = self.aux(x)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x, gen_p
    
    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x
    
    def forward_frag_cl(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2, aggregator):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        x = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim = 1)
        # x = (x1+x2)/2.0
        x = aggregator(x)
        x = self.projection_head(x)
        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2




    def loss_cl(self, x1, x2, frag1, frag2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                     torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                     torch.einsum('i,j->ij', x1_abs, frag2_abs)

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 + neg2 - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss



def train(loader, model, optimizer, device, old, aggregator):

    model.train()
    train_loss_accum = 0

    for step, (batch, batch1, batch2) in enumerate(loader):
        # _, batch1, batch2 = batch
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        
        if old == 1:
            x = model.forward_cl(batch.x, batch.edge_index,
                                batch.edge_attr, batch.batch)
            #x1 = model.forward_cl(batch1.x, batch1.edge_index,
            #                      batch1.edge_attr, batch1.batch)
            #x2 = model.forward_cl(batch2.x, batch2.edge_index,
            #                      batch2.edge_attr, batch2.batch)
            x2 = model.forward_frag_cl_old(batch1.x, batch1.edge_index,
                                batch1.edge_attr, batch1.batch,
                                batch2.x, batch2.edge_index,
                                batch2.edge_attr, batch2.batch)
            
            loss = model.loss_cl_old(x, x2)
        else:    
            x = model.forward_cl(batch.x, batch.edge_index,
                                batch.edge_attr, batch.batch)
            #x1 = model.forward_cl(batch1.x, batch1.edge_index,
            #                      batch1.edge_attr, batch1.batch)
            #x2 = model.forward_cl(batch2.x, batch2.edge_index,
            #                      batch2.edge_attr, batch2.batch)
            x2, frag1, frag2 = model.forward_frag_cl(batch1.x, batch1.edge_index,
                                batch1.edge_attr, batch1.batch,
                                batch2.x, batch2.edge_index,
                                batch2.edge_attr, batch2.batch, aggregator)
            
            loss = model.loss_cl(x, x2, frag1, frag2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum / (step + 1)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GraphFrag')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch')
    parser.add_argument('--decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100 , help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features across layers are combined.')
    parser.add_argument('--gnn_type', type=str, default="gin", help='gnn model type')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dataset', type=str, default=None, help='root dir of dataset')
    parser.add_argument('--num_layer', type=int, default=5, help='message passing layers')
    # parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset")
    parser.add_argument('--output_model_file', type=str, default='', help='model save path')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default='uniform')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    # parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_model_dir', type=str, default='')
    parser.add_argument('--n_mol', type=int, default=50000, help='number of unique smiles/molecules')
    parser.add_argument('--data_folder', type=str, default='../datasets')
    parser.add_argument('--choose', type=int, default=0)
    parser.add_argument('--old', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset
    if 'GEOM' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/GEOM_2D_nmol{}_cut_singlebond'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # set up model
    gnn = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn)
    aggregator = vits.vit_tiny().to(device)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(aggregator.parameters()), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        pretrain_loss = train(loader, model, optimizer, device, args.old, aggregator)
        print('Epoch: {:3d}\tLoss:{:.3f}\tTime: {:.3f}:'.format(
            epoch, pretrain_loss, time.time() - start_time))

    if not args.output_model_dir == '':
        print('save')
        saver_dict = {'model': model.state_dict()}
        print(args.output_model_dir + '_model_complete.pth')
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
        torch.save(model.gnn.state_dict(), args.output_model_dir + '_model.pth')
