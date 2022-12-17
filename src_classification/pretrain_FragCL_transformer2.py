import argparse

from loader_graphfrag import MoleculeDataset_aug_frag
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

from copy import deepcopy
import vision_transformer as vits


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


def train(loader, model, optimizer, device, aggregator):

    #dataset.aug = "none"
    #dataset1 = dataset.shuffle()
    #dataset2 = deepcopy(dataset1)
    #dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    #dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    #loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    #loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()
    train_loss_accum = 0

    for step, (batch, batch1, batch2, _, _, _) in enumerate(tqdm(loader, desc="Iteration")):
        # _, batch1, batch2 = batch
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        

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

        #if step % 100 == 0:
        #    print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())
    return train_loss_accum / (step + 1)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent_frag_single_bond2', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    
    
    parser.add_argument('--aug_mode', type=str, default='choose')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    # parser.add_argument('--gamma', type=float, default=0.1)
    #parser.add_argument('--output_model_dir', type=str, default='')
    #parser.add_argument('--n_mol', type=int, default=50000, help='number of unique smiles/molecules')
    #parser.add_argument('--data_folder', type=str, default='../datasets')
    parser.add_argument('--choose', type=int, default=0)

    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset = MoleculeDataset_aug_frag("dataset/" + args.dataset, dataset=args.dataset, choose=args.choose)

    #indices = [i for i in range(0,len(dataset), 40)]

    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    #dataset = torch.utils.data.Subset(dataset, indices)
    print(len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    num_gnn = 0
    for p in gnn.parameters():
        num_gnn += p.numel()        

    model = graphcl(gnn)
    aggregator = vits.vit_tiny().to(device)

    num_agg = 0
    for p in aggregator.parameters():
        num_agg += p.numel()
    
    print(num_gnn ,num_agg)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(aggregator.parameters()), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(loader, model, optimizer, device, aggregator)

        #print(train_acc)
        print(train_loss)

        if epoch % 10 == 0:
            torch.save(gnn.state_dict(), "./model/2FragCL_" + str(epoch) + ".pth")
            torch.save(aggregator.state_dict(), "./model/2FragCL_aggregator_" + str(epoch) + ".pth")

if __name__ == "__main__":
    main()
