import argparse
from textwrap import dedent
import time

import numpy as np
import torch
import torch.optim as optim
from models import GNN
from pretrain_JOAO import fragcl4
from torch_geometric.data import DataLoader

from datasets import MoleculeDatasetFrag4
from os.path import join

def train(loader, model, optimizer, device):

    model.train()
    train_loss_accum = 0

    for step, (batch, batch1, batch2, batch11, batch12, batch21, batch22) in enumerate(loader):
        # _, batch1, batch2 = batch
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch11 = batch11.to(device)
        batch12 = batch12.to(device)
        batch21 = batch21.to(device)
        batch22 = batch22.to(device)
        
        x = model.forward_all(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        f1 = model.forward_pool(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        proj_f1 = model.project(f1)
        f2 = model.forward_pool(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        proj_f2 = model.project(f2)
        f11 = model.forward_pool(batch11.x, batch11.edge_index, batch11.edge_attr, batch11.batch)
        proj_f11 = model.project(f11)
        f12 = model.forward_pool(batch12.x, batch12.edge_index, batch12.edge_attr, batch12.batch)
        proj_f12 = model.project(f12)
        f21 = model.forward_pool(batch21.x, batch21.edge_index, batch21.edge_attr, batch21.batch)
        proj_f21 = model.project(f21)
        f22 = model.forward_pool(batch22.x, batch22.edge_index, batch22.edge_attr, batch22.batch)
        proj_f22 = model.project(f22)

        # logit generation for original molecule x
        batch1_atoms = torch.bincount(batch1.batch)
        batch2_atoms = torch.bincount(batch2.batch)
        total_atoms = batch1_atoms + batch2_atoms
        ratio1 = batch1_atoms/total_atoms
        ratio2 = batch2_atoms/total_atoms
        f1_f2 = f1 * ratio1.unsqueeze(1).repeat(1, f1.size(1)) + f2 * ratio2.unsqueeze(1).repeat(1, f2.size(1))
        proj_f1_f2 = model.project(f1_f2)
        loss1 = model.loss_cl4(x, proj_f1_f2, proj_f1, proj_f2, proj_f11, proj_f12, proj_f21, proj_f22)

        # logit generation for fragment1
        batch1_atoms = torch.bincount(batch11.batch)
        batch2_atoms = torch.bincount(batch12.batch)
        total_atoms = batch1_atoms + batch2_atoms
        ratio1 = batch1_atoms/total_atoms
        ratio2 = batch2_atoms/total_atoms        
        f11_f12 = f11 * ratio1.unsqueeze(1).repeat(1, f11.size(1)) + f12 * ratio2.unsqueeze(1).repeat(1, f12.size(1))
        proj_f11_f12 = model.project(f11_f12)
        loss2 = model.loss_cl2(proj_f1, proj_f11_f12, proj_f11, proj_f12)

        # logit generation for fragment1
        batch1_atoms = torch.bincount(batch21.batch)
        batch2_atoms = torch.bincount(batch22.batch)
        total_atoms = batch1_atoms + batch2_atoms
        ratio1 = batch1_atoms/total_atoms
        ratio2 = batch2_atoms/total_atoms        
        f21_f22 = f21 * ratio1.unsqueeze(1).repeat(1, f21.size(1)) + f22 * ratio2.unsqueeze(1).repeat(1, f22.size(1))
        proj_f21_f22 = model.project(f21_f22)
        loss3 = model.loss_cl2(proj_f2, proj_f21_f22, proj_f21, proj_f22)

        loss = (loss1 + loss2 + loss3)/3

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
        root_2d = '{}/molecule_datasets/{}'.format(args.data_folder, args.dataset)
        dataset = MoleculeDatasetFrag4(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # set up model
    gnn = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = fragcl4(gnn)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        pretrain_loss = train(loader, model, optimizer, device)
        print('Epoch: {:3d}\tLoss:{:.3f}\tTime: {:.3f}:'.format(
            epoch, pretrain_loss, time.time() - start_time))

    if not args.output_model_dir == '':
        print('save')
        saver_dict = {'model': model.state_dict()}
        print(args.output_model_dir + '_model_complete.pth')
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
        torch.save(model.gnn.state_dict(), args.output_model_dir + '_model.pth')
