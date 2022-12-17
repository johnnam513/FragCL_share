import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from models import GNN, AutoEncoder, SchNet, VariationalAutoEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from util import dual_CL
from pretrain_JOAO import graphcl, graphcl3d
from datasets import Molecule3DDatasetFragRandomaug3dRandcut


def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(gnn.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(gnn.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                #'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                #'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    
    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    
    for step, (batch, batch1, batch2, orig_batch, orig_batch1, orig_batch2) in enumerate(l):
        #print('ddddd')
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        orig_batch.to(device)
        orig_batch1.to(device)
        orig_batch2.to(device)

        molecule_2D_repr = molecule_model_2D.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        molecule_2D_repr_mixed, frag_2D_repr1, frag_2D_repr2 = molecule_model_2D.weighted_forward_frag_cl(batch1.x, batch1.edge_index,
                                batch1.edge_attr, batch1.batch,
                                batch2.x, batch2.edge_index,
                                batch2.edge_attr, batch2.batch)       
        loss_2D = molecule_model_2D.weighted_loss_cl_onlyneg(molecule_2D_repr, molecule_2D_repr_mixed, frag_2D_repr1, frag_2D_repr2, batch1.batch, batch2.batch)

        if args.model_3d == 'schnet':
            molecule_3D_repr = molecule_model_3D.forward_cl(orig_batch.x[:, 0], orig_batch.positions + torch.randn_like(orig_batch.positions).cuda(), orig_batch.batch) #+ torch.randn_like(orig_batch.positions).cuda()
            molecule_3D_repr_mixed, frag_3D_repr1, frag_3D_repr2 = molecule_model_3D.weighted_forward_frag_cl(orig_batch1.x[:,0], orig_batch1.positions + torch.randn_like(orig_batch1.positions).cuda(), orig_batch1.batch, #+ torch.randn_like(orig_batch1.positions).cuda()
                                orig_batch2.x[:,0], orig_batch2.positions + torch.randn_like(orig_batch2.positions).cuda(), orig_batch2.batch)       #+ torch.randn_like(orig_batch2.positions).cuda()
            loss_3D = molecule_model_3D.weighted_loss_cl_onlyneg(molecule_3D_repr, molecule_3D_repr_mixed, frag_3D_repr1, frag_3D_repr2, orig_batch1.batch, orig_batch2.batch)            

        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        


        #AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
        #AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
        #AE_acc_1 = AE_acc_2 = 0
        #AE_loss = (AE_loss_1 + AE_loss_2) / 2

        CL_loss_accum += CL_loss.detach().cpu().item() #+ loss_2D.detach().cpu().item() + loss_3D.detach().cpu().item()
        CL_acc_accum += CL_acc
        #AE_loss_accum += AE_loss.detach().cpu().item()
        #AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2

        loss = 0
        if epoch < 10:
            loss += loss_2D + loss_3D
        else:
            loss += CL_loss + loss_2D + loss_3D
        #if args.alpha_2 > 0:
        #    loss += AE_loss * args.alpha_2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(molecule_model_2D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(molecule_model_3D.parameters(), 5)
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tAE Acc: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, AE_loss_accum, AE_acc_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/GEOM_2D_3D_nmol{}_cut_singlebond_nobranch'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug3dRandcut(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    molecule_model_2D = graphcl(gnn).to(device)
    #molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout)
        molecule_model_3D = graphcl3d(molecule_model_3D).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))

#    if args.AE_model == 'AE':
#        AE_2D_3D_model = AutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
#        AE_3D_2D_model = AutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
#    elif args.AE_model == 'VAE':
#        AE_2D_3D_model = VariationalAutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
#        AE_3D_2D_model = VariationalAutoEncoder(
#            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
#    else:
#        raise Exception

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})
#    model_param_group.append({'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.gnn_lr_scale})
#    model_param_group.append({'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch)

    save_model(save_best=False)
