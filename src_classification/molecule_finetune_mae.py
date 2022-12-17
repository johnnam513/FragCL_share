from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models import GNN, GNN_graphpred_mae, GNN_graphpred
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
from util import get_num_task
from models_mae import MaskedAutoencoderViT

from datasets import MoleculeDataset_mae
from pretrain_JOAO import graphcl_brics

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for step, (batch, batch0, batch1, batch2, batch3) in enumerate(loader):
        
        batch = batch.to(device)
        batch0 = batch0.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch3 = batch3.to(device)

        rep_2D_0 = model.molecule_model.forward_mae(batch0.x, batch0.edge_index, batch0.edge_attr, batch0.batch).unsqueeze(1)
        rep_2D_1 = model.molecule_model.forward_mae(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch).unsqueeze(1)
        rep_2D_2 = model.molecule_model.forward_mae(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch).unsqueeze(1)
        rep_2D_3 = model.molecule_model.forward_mae(batch3.x, batch3.edge_index, batch3.edge_attr, batch3.batch).unsqueeze(1)

        reps = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3], dim=1)
        #pred = model.do_mae(reps)

        #reps = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3], dim=1)
        reps = reps.mean(dim=1)
        #pred = model.do_mae(reps)
        #reps = model.molecule_model.forward_mae(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = model.graph_pred_linear(reps)
        #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for step, (batch, batch0, batch1, batch2, batch3) in enumerate(loader):
        batch = batch.to(device)
        batch0 = batch0.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch3 = batch3.to(device)
        with torch.no_grad():
            rep_2D_0 = model.molecule_model.forward_mae(batch0.x, batch0.edge_index, batch0.edge_attr, batch0.batch).unsqueeze(1)
            rep_2D_1 = model.molecule_model.forward_mae(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch).unsqueeze(1)
            rep_2D_2 = model.molecule_model.forward_mae(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch).unsqueeze(1)
            rep_2D_3 = model.molecule_model.forward_mae(batch3.x, batch3.edge_index, batch3.edge_attr, batch3.batch).unsqueeze(1)

            reps = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3], dim=1)
            #pred = model.do_mae(reps)

            #reps = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3], dim=1)
            reps = reps.mean(dim=1)
            #pred = model.do_mae(reps)
            #reps = model.molecule_model.forward_mae(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.graph_pred_linear(reps)
            
            #reps = torch.cat([rep_2D_0, rep_2D_1, rep_2D_2, rep_2D_3], dim=1)
            #reps = reps.mean(dim=1)
            ##reps = model.molecule_model.forward_mae(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            #pred = model.graph_pred_linear(reps)
            
            #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            #reps, _, _ = model.mae.forward_encoder(reps, 0.0)
            #print(reps.mean(dim=1))

            #pred = model.do_mae(reps)

        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print('{} is invalid'.format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = '../datasets/molecule_datasets/'
    dataset = MoleculeDataset_mae(dataset_folder + args.dataset, dataset=args.dataset)
    print(dataset)

    eval_metric = roc_auc_score

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type)
    molecule_model = graphcl_brics(molecule_model)

    mae = MaskedAutoencoderViT(patch_size=16, in_chans=1,
                 embed_dim=300, depth=16, num_heads=10,
                 decoder_embed_dim=128, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False).to(device)
    model = GNN_graphpred_mae(args=args, num_tasks=num_tasks,
                          molecule_model=molecule_model, mae=mae)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    # set up optimizer
    # different learning rates for different parts of GNN
    model_param_group = [{'params': model.molecule_model.parameters()},
                         {'params': model.mae.parameters()},
                         {'params': model.graph_pred_linear.parameters(),
                          'lr': args.lr * args.lr_scale}]
    
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_roc, train_acc, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_roc = train_acc = 0
        val_roc, val_acc, val_target, val_pred = eval(model, device, val_loader)
        test_roc, test_acc, test_target, test_pred = eval(model, device, test_loader)

        train_roc_list.append(train_roc)
        train_acc_list.append(train_acc)
        val_roc_list.append(val_roc)
        val_acc_list.append(val_acc)
        test_roc_list.append(test_roc)
        test_acc_list.append(test_acc)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1
            if not args.output_model_dir == '':
                output_model_path = join(args.output_model_dir, 'model_best.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)

                filename = join(args.output_model_dir, 'evaluation_best.pth')
                np.savez(filename, val_target=val_target, val_pred=val_pred,
                         test_target=test_target, test_pred=test_pred)

    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))

    if args.output_model_dir is not '':
        output_model_path = join(args.output_model_dir, 'model_final.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)
