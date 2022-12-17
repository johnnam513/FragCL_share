import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from models import GNN
from pretrain_JOAO import graphcl
from torch_geometric.data import DataLoader

from datasets import Molecule3DDatasetFragRandomaugDouble
from os.path import join

def train(loader, model, optimizer, device):

    model.train()
    train_loss_accum = 0

    for step, (first_batch, first_batch1, first_batch2, second_batch, second_batch1, second_batch2) in enumerate(loader):
        # _, batch1, batch2 = batch
        first_batch = first_batch.to(device)
        first_batch1 = first_batch1.to(device)
        first_batch2 = first_batch2.to(device)
        
        second_batch = second_batch.to(device)
        second_batch1 = second_batch1.to(device)
        second_batch2 = second_batch2.to(device)
        


        first_x = model.forward_cl(first_batch.x, first_batch.edge_index,
                              first_batch.edge_attr, first_batch.batch)
        #x1 = model.forward_cl(batch1.x, batch1.edge_index,
        #                      batch1.edge_attr, batch1.batch)
        #x2 = model.forward_cl(batch2.x, batch2.edge_index,
        #                      batch2.edge_attr, batch2.batch)
        first_x2 = model.forward_frag_cl(first_batch1.x, first_batch1.edge_index,
                              first_batch1.edge_attr, first_batch1.batch,
                              first_batch2.x, first_batch2.edge_index,
                              first_batch2.edge_attr, first_batch2.batch)
        
        #print(first_x[:,:200].shape)
        loss1 = model.loss_cl(first_x[:,:200], first_x2[:,:200])


        second_x = model.forward_cl(second_batch.x, second_batch.edge_index,
                              second_batch.edge_attr, second_batch.batch)
        #x1 = model.forward_cl(batch1.x, batch1.edge_index,
        #                      batch1.edge_attr, batch1.batch)
        #x2 = model.forward_cl(batch2.x, batch2.edge_index,
        #                      batch2.edge_attr, batch2.batch)
        second_x2 = model.forward_frag_cl(second_batch1.x, second_batch1.edge_index,
                              second_batch1.edge_attr, second_batch1.batch,
                              second_batch2.x, second_batch2.edge_index,
                              second_batch2.edge_attr, second_batch2.batch)
        
        loss2 = model.loss_cl(second_x[:,100:], second_x2[:,100:])

        loss = loss1 + loss2
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
        root_2d = '{}/GEOM_2D_nmol{}_cut_CCsinglebond'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaugDouble(root=root_2d, n_mol=n_mol, choose=args.choose,
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
